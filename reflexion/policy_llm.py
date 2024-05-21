import sys
import gradio as gr
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    get_peft_config
)
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np
import transformers
from critic import Critic
from torch.distributions.categorical import Categorical
from langchain_community.llms import HuggingFacePipeline
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class LLMAgent(nn.Module):
    def __init__(self, max_token=100, normalization_mode = 'token', load_path = None, load_8bit = True):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = '/scratch2/nlp/plm/Meta-Llama-3-8B-Instruct'
        self.lora_r  = 8
        self.lora_alpha = 16
        self.lora_dropout = 0
        self.lora_target_modules  = ["q_proj", "v_proj",]
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
        self.tokenizer.pad_token_id = (0)
        self.llama = self._init_llama()

        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)

    def _init_llama(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_8bit=self.load_8bit,
            device_map="auto",
            cache_dir=os.path.join(root, 'weights/llama')
        )
        #model.gradient_checkpointing_enable()


        # if not self.load_8bit:
        #     model.half().to(self.device)
        # else:
        #     model = prepare_model_for_kbit_training(model) #, use_gradient_checkpointing=True


        return model

    def _init_actor(self, lora_weights = None):
        if lora_weights is None:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            #peft_config = get_peft_config(config)
            
            model = get_peft_model(self.llama, config)
            #model = prepare_model_for_kbit_training(model) #, use_gradient_checkpointing=True

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.llama,
                lora_weights,
                torch_dtype=torch.float16,
                load_in_8bit=self.load_8bit,
                device_map="auto"
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
            
        return model

    def _init_critic(self, critic_weights = None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head_mlp3.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic


    def get_model(self, prompt, k_sent=1):
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.actor,
            tokenizer=self.tokenizer,
            repetition_penalty=1.1,
            min_new_tokens = 30,
            max_new_tokens = self.max_token,
            temperature=1.5,
            device_map="cuda")
        # llm = HuggingFacePipeline(pipeline=query_pipeline)

        sequences = pipeline(
            prompt,
            do_sample=True,
            top_k = 30,
            #top_p=0.85,
            num_return_sequences= k_sent,  #https://zhuanlan.zhihu.com/p/643949567, https://zhuanlan.zhihu.com/p/653926703
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if k_sent <= 1:
            return sequences[0]['generated_text'].split(prompt)[1].split('\n')[0]
        else:
            return [i['generated_text'].split(prompt)[1].split('\n')[0] for i in sequences ]
    
    def save(self, epoch, exp_path):
        print("save model")
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        #logger.info(list(self.agent.actor.parameters()))
        #logger.info(list(filter(lambda p: p.requires_grad, self.actor.parameters())))
        self.actor.save_pretrained(exp_path)  # safe_serialization=False
        # save critic
        torch.save(self.critic.v_head_mlp3.state_dict(), os.path.join(exp_path, "critic.pth"))

    def load(self, exp_path):
        print("----------------load model")
        lora_weights = exp_path
        critic_weights = os.path.join(exp_path, "critic.pth")
        self.actor = self._init_actor(lora_weights).to(self.device)
        self.critic = self._init_critic(critic_weights).to(self.device)
    
    def get_value(self, x): ##x is a list
            
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"]
        
        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, obs, actions, action=None, is_warmup=False, return_value = True):

        prompt = [obs[0]]
        action_list = [actions]
        action_num = len(action_list[0])

        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]

        # print(sequence)
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        
        attention_mask = inputs["attention_mask"]
        if is_warmup:
            with torch.no_grad():
                outputs = self.actor(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.actor(input_ids, attention_mask=attention_mask)
        
        action_list = [item for sublist in action_list for item in sublist]
        self.action_list_ids = self.tokenizer(action_list, return_tensors="pt", padding=True)

        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim = -1) - 1 #delete first token
        sequence_length = torch.sum(attention_mask, dim = -1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        # maybe no need to use it, directly use logits
        logits = torch.log_softmax(outputs.logits, dim=-1)

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

        slices = [gen_logits[i, start-1:end-1] for i, (start, end) in enumerate(action_index)]
        
        action_logits = torch.stack([torch.sum(s) for s in slices])
        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits


        action_logits = action_logits.reshape(-1, action_num).float()

        probs = Categorical(logits=action_logits)
        if action is None:
            action = probs.sample()

        if return_value:
            return action, probs.log_prob(action), probs.entropy(), self.get_value(prompt)
        else:
            return action, probs.log_prob(action), probs.entropy(), None

    def format_step(step: str) -> str:
        return step.strip('\n').strip().replace('\n\n', ' ').replace('\n', ' ').replace('\'', '')