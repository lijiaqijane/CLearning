import sys, os, csv, json
from time import time
sys.path.append('..')
root = '../root/'
import warnings
warnings.filterwarnings("ignore")
import os
import csv
import json
import os
from time import time
from datasets import load_dataset
import random
import os
from torch.utils.tensorboard import SummaryWriter
from action import ReactAgent, logger
from transformers import  AutoTokenizer
from ppo_llm_pomdp import Policy

def get_max_obslen(ds):
    tokenizer = AutoTokenizer.from_pretrained('/scratch2/nlp/plm/Meta-Llama-3-8B-Instruct') #Meta-Llama-3-8B-Instruct, Llama-2-13b-chat-hf
    max_seq_len = 0
    for i in range(len(ds)):
        curr_q = ds[i]['question']
        length = len(tokenizer(curr_q, return_tensors="pt")["input_ids"][0])
        if length >= max_seq_len:
            max_seq_len = length
    return max_seq_len



#task = str(['place_plant', 'collect_wood', 'place_table','make_wood_sword', 'make_wood_pickaxe', 'eat_plant', 'collect_coal', 'collect_stone', 'place_stone','place_furnace', 'make_stone_sword', 'make_stone_pickaxe', 'collect_iron', 'make_iron_sword','make_iron_pickaxe', 'collect_diamond'])
task = 'collect_wood'
num_updates = 2000   ##??最大步数
 
# writer = SummaryWriter(f"../writer/test")
policy = Policy(max_obs = 200)  
global_step, no_q  = 0, 0

for update in range(1, num_updates + 1):
    total_reward, hits = 0, 0
    logger.info('===========Current train update: '+str(update))
    # no_seed = random.randint(1,len(task_list))
    # task =  task_list[0]   
    #logger.info('==========='+str(task))

    frac = 1.0 - (update - 1.0) / num_updates
    global_step, reward, hits = policy.trainer(task, global_step, frac, hits, writer=None)
            

    if global_step // 500 > 0 : 
        policy.agent.save(global_step // 500, "../result/")

    total_reward += reward
    logger.info('===========Current step: {},  =====total_reward: {}, =====hits: {}'.format(global_step, total_reward, hits))
    no_q += 1

# writer.close()



