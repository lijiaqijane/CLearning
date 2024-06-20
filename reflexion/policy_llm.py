import sys
import torch
from peft import (
    LoraConfig,
    get_peft_model
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
import os
import torch.nn as nn
import numpy as np
import transformers
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pad_sequence

from .llama3_formatter import ChatFormat, Dialog, LLama3Tokenizer
from .critic import Critic

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
        self.base_model = '/scratch2/nlp/plm/Meta-Llama-3-8B-Instruct' #Meta-Llama-3-8B-Instruct, Llama-2-7b-chat-hf, Llama-2-13b-chat-hf
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
        self.llama_formatter = ChatFormat(LLama3Tokenizer(f'{self.base_model}/tokenizer.json'))

        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)

    def _init_llama(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_8bit=self.load_8bit,
            device_map={'':0},
            cache_dir=os.path.join(root, 'weights/llama')
        )
        #model.gradient_checkpointing_enable()
        # if not self.load_8bit:
        #     model.half().to(self.device)
        # else:
        #     model = prepare_model_for_kbit_training(model) #, use_gradient_checkpointing=True


        return model

    def _init_actor(self, lora_weights = None) -> PeftModel:
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
                load_in_8bit=self.load_8bit,
                device_map={'':0}
            )
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
            model.print_trainable_parameters()
            #print("loadpeft_savedmodels")
            
        # if torch.__version__ >= "2" and sys.platform != "win32":
        #     model = torch.compile(model)
            
        return model

    def _init_critic(self, critic_weights = None) -> Critic:
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head_mlp3.load_state_dict(torch.load(critic_weights, map_location= "cpu")) #!critic.v_head.load_state_dict(torch.load(critic_weights, map_location= "cpu"))
        return critic


    def get_model(self, prompt, k_sent=1):
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.actor,
            tokenizer=self.tokenizer,
            repetition_penalty=1.1,
            min_new_tokens = 30,
            max_new_tokens = self.max_token,
            temperature= 0.1,
            device_map={'':0})
        # llm = HuggingFacePipeline(pipeline=query_pipeline)

        sequences = pipeline(
            prompt,
            do_sample=True,
            # top_k = 30,
            #top_p=0.85,
            num_return_sequences= k_sent,  #https://zhuanlan.zhihu.com/p/643949567, https://zhuanlan.zhihu.com/p/653926703
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if k_sent <= 1:
            return [sequences[0]['generated_text'].split(prompt)[1].strip('\n').split(']')[0].replace('[','')]
        else:
            return [i['generated_text'].split(prompt)[1].split('\n')[0] for i in sequences ]


#     def get_model(self, prompt, k_sent=1):
#         pipeline = transformers.pipeline(
#             "text-generation",
#             model=self.actor,
#             tokenizer=self.tokenizer,
#             repetition_penalty=1.1,
#             min_new_tokens = 30,
#             max_new_tokens = self.max_token,
#             temperature=0.1,
#             device_map={'':0})
#         # llm = HuggingFacePipeline(pipeline=query_pipeline)

#         cand = []
#         for i in [' Thought: ',' Ask: ',' Search: ',' Act: ']:
#             if i !=' Act: ':
#                 prompt += i
#                 sequences = pipeline(
#                     prompt,
#                     do_sample=True,
#                     top_k = 1,
#                     #top_p=0.85,
#                     num_return_sequences= k_sent,  #https://zhuanlan.zhihu.com/p/643949567, https://zhuanlan.zhihu.com/p/653926703
#                     eos_token_id=self.tokenizer.eos_token_id,
#                 )
#                 cand.append(i+sequences[0]['generated_text'].split(prompt)[1].split('\n')[0])
#             else:
#                 act_list = ["0. Noop: Always applicable.","1. Move West: Flat ground west of the agent.","2. Move East: Flat ground east of the agent.",\
# "3. Move North: Flat ground north of the agent.","4. Move South: Flat ground south of the agent.","5. Do: Facing creature or material; have necessary tool.",\
# "6. Sleep: Energy level is below maximum.","7. Place Stone: Stone in inventory.","8. Place Table: Wood in inventory.","9. Place Furnace: Stone in inventory.",\
# "10. Place Plant: Sapling in inventory.","11. Make Wood Pickaxe: Nearby table; wood in inventory.","12. Make Stone Pickaxe: Nearby table; wood, stone in inventory.",\
# "13. Make Iron Pickaxe: Nearby table, furnace; wood, coal, iron an inventory.","14. Make Wood Sword: Nearby table; wood in inventory.",\
# "15. Make Stone Sword: Nearby table; wood, stone in inventory.","16. Make Iron Sword: Nearby table, furnace; wood, coal, iron in inventory."]
#                 no_seed = random.randint(0,len(act_list))
#                 cand.append(i+ act_list[no_seed].split('.')[1].split(':')[0] +',' + act_list[no_seed].split('.')[0])
#         return cand

    
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
    
    def generate(self, prompt):
        # Which to use?
        return self.actor.generate(prompt, assistant_model=self.llama, prompt_lookup_num_tokens = 3)
        # return self.llama.generate(prompt, prompt_lookup_num_tokens = 3)

    def get_value(self, input_ids): ##x is a list            
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != 0).long()
        
        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, obs: Dialog, prompt: Dialog, actions, action=None, is_warmup=False, return_value = True):
        prompt_ids =  self.llama_formatter.encode_dialog_prompt(prompt)
        raw_input_ids = []
        for act in actions:
            raw_input_ids.append(prompt_ids + self.tokenizer.encode(act.strip(), bos=False, eos=False))
        input_ids = pad_sequence(raw_input_ids, batch_first=True, padding_value=0)
        attention_mask = (input_ids != 0).long()

        if is_warmup:
            with torch.no_grad():
                outputs = self.actor(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.actor(input_ids, attention_mask=attention_mask)

        #???
        action_list = [actions]
        action_num = len(action_list[0])
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

        obs_ids = [self.llama_formatter.encode_dialog_prompt(prompt)]
        if return_value:
            return action, probs.log_prob(action), probs.entropy(), self.get_value(obs)
        else:
            return action, probs.log_prob(action), probs.entropy(), None

    def format_step(step: str) -> str:
        return step.strip('\n').strip().replace('\n\n', ' ').replace('\n', ' ').replace('\'', '')
