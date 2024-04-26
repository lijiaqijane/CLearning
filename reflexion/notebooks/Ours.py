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
import openai
import csv
from time import time
import json
import os
from time import time
import openai
import csv
import os
import csv
from datasets import load_dataset
import os

from util import summarize_react_trial, log_react_trial, save_agents
from action import ReactAgent, logger

from ppo_llm_pomdp import Policy


q, a=  "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"
policy = Policy()
pre_global_step, global_step = 0, 0
num_updates = 1
for update in range(1, num_updates + 1):
    
    frac = 1.0 - (update - 1.0) / num_updates
    global_step = policy.trainer(q, a, global_step, frac)
            
    if global_step // 10000 != pre_global_step // 10000: 
        agent.save(global_step // 10000, "./")
    pre_global_step = global_step
    logger.info('Current step: '+str(global_step))
