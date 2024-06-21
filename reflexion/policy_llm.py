import sys
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
import os
import torch.nn as nn
import numpy as np
import transformers
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence

from .llama3_formatter import ChatFormat, Dialog
from .critic import Critic

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LLMAgent(nn.Module):
    def __init__(
        self,
        max_obs,
        action_list,
        max_token=100,
        normalization_mode="token",
        load_path=None,
        load_8bit=True,
    ):
        super().__init__()

        self.obs_length = max_obs

        self.load_8bit = load_8bit
        self.base_model = "/scratch2/nlp/plm/Meta-Llama-3-8B-Instruct"  # Meta-Llama-3-8B-Instruct, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0
        self.lora_target_modules = [
            "q_proj",
            "v_proj",
        ]
        self.max_token = max_token
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:
            pass

        self.normalization_mode = normalization_mode
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = 0
        self.llama = self._init_llama()

        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)

        self.action_list = action_list
        action_list_ids = self.tokenizer(
            self.action_list, return_tensors="pt", padding=True
        )
        self.action_list_length = (
            torch.sum(action_list_ids["attention_mask"], dim=-1) - 1
        ).to(self.device)
        self.action_word_num = torch.tensor(
            [len(action.split()) for action in self.action_list]
        ).to(self.device)


    def _init_llama(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map={"": 0},
            cache_dir=os.path.join(root, "weights/llama"),
        )
        # model.gradient_checkpointing_enable()
        if not self.load_8bit:
            model.half().to(self.device)
        else:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

        return model

    def _init_actor(self, lora_weights=None) -> PeftModel:
        if lora_weights is None:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.llama, config)
            model.print_trainable_parameters()

            # old_state_dict = model.state_dict
            # model.state_dict = (
            #     lambda self, *_, **__: get_peft_model_state_dict(
            #         self, old_state_dict()
            #     )
            # ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.llama,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
            for name, param in model.named_parameters():
                if "lora" in name:
                    print("unfreezing", name)
                    param.requires_grad = True
            model.print_trainable_parameters()
            # print("loadpeft_savedmodels")

        if not self.load_8bit:
            model.half().to(self.device)
        else:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=True
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model, dynamic=True)

        return model

    def _init_critic(self, critic_weights=None) -> Critic:
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            # critic.v_head_mlp3.load_state_dict(
            #     torch.load(critic_weights, map_location="cpu")
            # )  #!critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
            critic.v_head_mlp3.load_state_dict(
                torch.load(critic_weights, map_location="cuda")
            )
        return critic

    @DeprecationWarning
    def get_model(self, prompt, k_sent=1):
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.actor,
            tokenizer=self.tokenizer,
            repetition_penalty=1.1,
            min_new_tokens=30,
            max_new_tokens=self.max_token,
            temperature=0.1,
            device_map={"": 0},
        )
        # llm = HuggingFacePipeline(pipeline=query_pipeline)

        sequences = pipeline(
            prompt,
            do_sample=True,
            # top_k = 30,
            # top_p=0.85,
            num_return_sequences=k_sent,  # https://zhuanlan.zhihu.com/p/643949567, https://zhuanlan.zhihu.com/p/653926703
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if k_sent <= 1:
            return [
                sequences[0]["generated_text"]
                .split(prompt)[1]
                .strip("\n")
                .split("]")[0]
                .replace("[", "")
            ]
        else:
            return [
                i["generated_text"].split(prompt)[1].split("\n")[0] for i in sequences
            ]

    def save(self, epoch, exp_path):
        print("save model")
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        # logger.info(list(self.agent.actor.parameters()))
        # logger.info(list(filter(lambda p: p.requires_grad, self.actor.parameters())))
        self.actor.save_pretrained(exp_path)  # safe_serialization=False
        # save critic
        torch.save(
            self.critic.v_head_mlp3.state_dict(), os.path.join(exp_path, "critic.pth")
        )

    def load(self, exp_path):
        print("----------------load model")
        lora_weights = exp_path
        critic_weights = os.path.join(exp_path, "critic.pth")
        self.actor = self._init_actor(lora_weights).to(self.device)
        self.critic = self._init_critic(critic_weights).to(self.device)

    def generate(self, prompt):
        # Which to use?
        return self.actor.generate(
            prompt, assistant_model=self.llama, prompt_lookup_num_tokens=3
        )
        # return self.llama.generate(prompt, prompt_lookup_num_tokens = 3)

    def get_value(self, x):  ##x is a list
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"]

        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(
        self,
        # obs: Dialog,
        prompt_str: str,
        action=None,
        is_warmup=False,
        return_value_and_info=True,
    ):
        sequence = [f"{prompt_str}{act}" for act in self.action_list]
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad() if is_warmup else torch.enable_grad():
            outputs = self.actor(input_ids, attention_mask=attention_mask)

        sequence_length = torch.sum(attention_mask, dim=-1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        # maybe no need to use it, directly use logits
        logits = torch.log_softmax(outputs.logits, dim=-1)

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

        slices = [
            gen_logits[i, start - 1 : end - 1]
            for i, (start, end) in enumerate(action_index)
        ]
        action_logits = torch.stack([torch.sum(s) for s in slices])

        if self.normalization_mode == "token":
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == "word":
            action_logits = action_logits / self.action_word_num
        elif self.normalization_mode == "sum":
            action_logits = action_logits

        action_logits = action_logits.reshape(-1, len(self.action_list)).float()

        probs = Categorical(logits=action_logits)

        if action is None:
            action = probs.sample()

        # obs_str = self.llama_formatter.format_dialog_prompt(prompt)
        if return_value_and_info:
            prompt_tensor = self.tokenizer(
                prompt_str,
                return_tensors="pt",
                padding="max_length",
                max_length=self.obs_length,
            )["input_ids"].to(self.device)
            return (
                action,
                probs.log_prob(action),
                probs.entropy(),
                self.get_value([prompt_str]),
                prompt_tensor,
            )
        else:
            return action, probs.log_prob(action), probs.entropy(), None

    def format_step(step: str) -> str:
        return (
            step.strip("\n")
            .strip()
            .replace("\n\n", " ")
            .replace("\n", " ")
            .replace("'", "")
        )
