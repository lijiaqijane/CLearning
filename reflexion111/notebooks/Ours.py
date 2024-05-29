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

def get_achievement(pre_ach, ach):
    new_ach = {}
    #pre_ach, ach = json.loads(pre_ach), json.loads(ach)
    for i in ach:
        if ach[i] > pre_ach[i]:
            new_ach[i] = ach[i] - pre_ach[i]
    return new_ach
        
        
#task = str(['place_plant', 'collect_wood', 'place_table','make_wood_sword', 'make_wood_pickaxe', 'eat_plant', 'collect_coal', 'collect_stone', 'place_stone','place_furnace', 'make_stone_sword', 'make_stone_pickaxe', 'collect_iron', 'make_iron_sword','make_iron_pickaxe', 'collect_diamond','collect_drink','collect_sapling','defeat_skeleton','defeat_zombie','eat_cow','wake_up'])
task = 'eat_cow'
num_updates = 1000   ##??最大步数
 
# writer = SummaryWriter(f"../writer/test")
policy = Policy(max_obs = 200)  
global_step, no_q  = 0, 0

for update in range(1, num_updates + 1):
    logger.info('===========Current train update: '+str(update))
    # no_seed = random.randint(1,len(task_list))
    # task =  task_list[0]   
    #logger.info('==========='+str(task))

    pre_achievement = {'collect_coal': 0, 'collect_diamond': 0, 'collect_drink': 0, 'collect_iron': 0,
                             'collect_sapling': 0, 'collect_stone': 0, 'collect_wood': 0, 'defeat_skeleton': 0,
                             'defeat_zombie': 0, 'eat_cow': 0, 'eat_plant': 0, 'make_iron_pickaxe': 0,
                             'make_iron_sword': 0, 'make_stone_pickaxe': 0, 'make_stone_sword': 0,
                             'make_wood_pickaxe': 0, 'make_wood_sword': 0, 'place_furnace': 0, 'place_plant': 0,
                             'place_stone': 0, 'place_table': 0, 'wake_up': 0}

    frac = 1.0 - (update - 1.0) / num_updates
    global_step, reward, achievement= policy.trainer(task, global_step, frac, writer=None)

    if global_step // 500 > 0 : 
        policy.agent.save(global_step // 500, "../result/")


    if  pre_achievement != achievement:
        hits = get_achievement(pre_achievement, achievement)
        logger.info('=====hits: {}'.format(hits))
        logger.info('=====curr_ach: {}'.format(achievement))
        pre_achievement = achievement
    logger.info('===========Current_step: {},  =====total_reward: {}'.format(global_step, reward))
    no_q += 1


logger.info('=====Final_ach: {}'.format(achievement))
# writer.close()



