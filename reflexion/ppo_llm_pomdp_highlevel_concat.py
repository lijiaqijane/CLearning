# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import gym
import crafter
import torch.optim as optim
from torch.distributions.categorical import Categorical
from policy_llm import LLMAgent
from torch.utils.tensorboard import SummaryWriter
import argparse
import pathlib
from Crafter.crafter.api.envWrapper import *
from Crafter.crafter.api.controller import *
import re
import datetime 
import os
import psutil
import gc
import logging
logger = logging.getLogger()
logger.setLevel('INFO')
formatter = logging.Formatter()
chlr = logging.StreamHandler() 
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  
fhlr = logging.FileHandler('Log.log')
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)
import sys
log_print = open('Error.log', 'w')
sys.stdout = log_print
sys.stderr = log_print
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):

    def __init__(self, max_steps, max_obs, curr_ckpt):
        super().__init__()

        self.num_steps = max_steps
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.policy_num_minibatches = self.num_steps
        self.value_num_minibatches = self.num_steps
        self.update_epochs = 1
        self.batch_size = self.num_steps
        self.policy_minibatch_size = int(self.batch_size // self.policy_num_minibatches)
        self.value_minibatch_size = int(self.batch_size // (self.value_num_minibatches*0.25))
        self.seed = 1
        self.cuda = True
        self.policy_learning_rate = 5e-6
        self.value_learning_rate = 1e-5
        self.norm_adv = False
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.gradient_checkpointing_steps = 8
        self.resume = True
        self.load_path = "/scratch/nlp/lijiaqi/CLearning/reflexion/result1/"+curr_ckpt
        self.normalization_mode = "word"
        self.actionlist = ['do nothing',
                        'attack the zombie',
                        'attack the skeleton',
                        'chop the tree',
                        'drink water',
                        'move to the diamond',
                        'move to the lava',
                        'place the stone',
                        'place the table',
                        'place the furnace',
                        'place the plant',
                        'craft the wood pickaxe',
                        'craft the stone pickax',
                        'craft the iron pickax',
                        'craft the wood sword',
                        'craft the stone sword',
                        'craft the iron sword',
                        'move to the grass',
                        'move to the plant',
                        'move to the stone',
                        'move to the table',
                        'move to the furnace',
                        'move to the cow',
                        'move to the zombie',
                        'move to the skeleton',
                        'move to the water',
                        'move to the sand',
                        'move to the tree',
                        'move to the coal',
                        'move to the iron']
        self.action_conv = {'do nothing':'noop', 
        'attack the zombie':'interactWithBlock',
        'attack the skeleton':'interactWithBlock',
        'chop the tree':'interactWithBlock',
        'drink water':'interactWithBlock',
        'place the stone':'place_stone',
        'place the table':'place_table',
        'place the furnace':'place_furnace',
        'place the plant':'place_plant',
        'craft the wood pickaxe':'make_wood_pickaxe', 
        'craft the stone pickax':'make_stone_pickaxe',
        'craft the iron pickax':'make_iron_pickaxe',
        'craft the wood sword':'make_wood_sword',
        'craft the stone sword':'make_stone_sword',
        'craft the iron sword':'make_iron_sword',
        'move to the grass':"getToBlock('grass')",
        'move to the plant':"getToBlock('plant')",
        'move to the stone':"getToBlock('stone')",
        'move to the table':"getToBlock('table')",
        'move to the furnace':"getToBlock('furnace')",
        'move to the cow':"getToBlock('cow')",
        'move to the zombie':"getToBlock('zombie')",
        'move to the skeleton':"getToBlock('skeleton')",
        'move to the water':"getToBlock('water')",
        'move to the sand':"getToBlock('sand')",
        'move to the tree':"getToBlock('tree')",
        'move to the coal':"getToBlock('coal')",
        'move to the iron':"getToBlock('iron')",
        'move to the diamond':"getToBlock('diamond')",
        'move to the lava':"getToBlock('lava')"}
        self.prompt = ""
        self.user = "You are playing game Crafter. You have unlock <'collect_drink','collect_sapling','collect_wood','place_plant>. To further finish the achievements < Collect Coal, Collect Diamond, Collect Drink, Collect Iron, Collect Sapling, Collect Stone, Collect Wood, kill Skeleton, kill Zombie, kill Cow, Eat Plant, Make Iron Pickaxe, Make Iron Sword, Make Stone Pickaxe, Make Stone Sword, Make Wood Pickaxe, Make Wood Sword, Place Furnace, Place Plant, Place Stone, Place Table, Wake Up >, you should first do:"


        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cuda else "cpu")


        if self.resume:
            self.agent = LLMAgent(normalization_mode=self.normalization_mode, load_path=self.load_path, load_8bit=True)
        else:
            self.agent = LLMAgent(normalization_mode=self.normalization_mode, load_8bit=True)
        
        #logger.info(list(self.agent.actor.parameters()))
        #logger.info(list(filter(lambda p: p.requires_grad, self.agent.actor.parameters())))
        self.policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.policy_learning_rate, eps=1e-5, weight_decay=0)
        self.value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.agent.critic.parameters()), lr=self.value_learning_rate, eps=1e-5)
            
        # ALGO Logic: Storage setup
        self.obs_length = max_obs
        self.obs = torch.zeros((self.num_steps, 1)+ (self.obs_length,)).to(self.device) ##+ envs.single_observation_space.shape
        self.actions = torch.zeros((self.num_steps, 1)).to(self.device)  ##+ envs.single_action_space.shape
        self.logprobs = torch.zeros((self.num_steps,1)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, 1)).to(self.device)
        self.dones = torch.zeros((self.num_steps, 1)).to(self.device)
        self.values = torch.zeros((self.num_steps, 1)).to(self.device)
        self.steps = torch.zeros((self.num_steps, 1)).to(self.device)
        self.candidate_action_num = 1


    def cutoff_obs(self, traj):
        if len(traj) > 1:
            traj.pop(0)
        return traj
    
    def controller_steps(self, env, action, tag=False):
        bot = AgentController(env)
        env_wrap = envWrapper(env)

        if action.startswith('ACTION: '):
            action = action.split(': ')[1]
            try:
                if '(' not in action:
                    action = f"{action}()"
                elif ',' in action:
                    arg = action.split('(')[1].split(',')[0].strip(')')
                    action = action.split('(')[0]+"('"+arg+"'," + action.split(',')[1]
                    action = action.replace("''","'")
                else:
                    arg = action.split('(')[1].strip(')')
                    action = action.split('(')[0]+"('"+arg+"')" 
                    action = action.replace("''","'")
                
                logger.info('execute: '+action)
                exec(f"bot.{action}")

                # if ('getToBlock' in action) or ('exploreDirection' in action):
                #      exec(f"bot.interactWithBlock()")

                observation = env_wrap.describe_frame()
                tag=True
            except:
                observation = "The generated action is not valid. Please check the available actions."

        elif action.startswith('THINK: '):
            observation = ''
        else:
            observation ="The output is not recognized."

        done = env.info['done']
        achievements =env.info['achievements']
        reward = env.info['reward']

        return tag, observation, reward, done, achievements

    def get_gpu_mem_info(self, gpu_id=0):
        import pynvml
        pynvml.nvmlInit()
        if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
            print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
            return 0, 0, 0

        handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
        total = round(meminfo.total / 1024 / 1024, 2)
        used = round(meminfo.used / 1024 / 1024, 2)
        free = round(meminfo.free / 1024 / 1024, 2)
        return total, used, free


    def get_cpu_mem_info(self):
        mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
        mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
        mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
        return mem_total, mem_free, mem_process_used


    def trainer(self, task, step_cnt, frac,  writer, is_warmup = False):  ##？？做对提前结束

        boolean = lambda x: bool(['False', 'True'].index(x))
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--area', nargs=2, type=int, default=(64, 64))
        parser.add_argument('--view', type=int, nargs=2, default=(9, 9))
        parser.add_argument('--length', type=int, default=None)
        parser.add_argument('--health', type=int, default=9)
        parser.add_argument('--window', type=int, nargs=2, default=(600, 600))
        parser.add_argument('--size', type=int, nargs=2, default=(0, 0))
        parser.add_argument('--load_world', type=pathlib.Path, default="/scratch/nlp/lijiaqi/CLearning/reflexion/Crafter/default")
        parser.add_argument('--model', type=str, default="ReAct_with_coordinate")
        parser.add_argument('--fps', type=int, default=3)
        parser.add_argument('--wait', type=boolean, default=False)
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--gen_world', type=boolean, default=False)
        parser.add_argument('--curr_update', type=int, default=0)
        args = parser.parse_args()

        env = crafter.Env(seed= args.seed, args = args)
        env.reset()

        observation = self.controller_steps(env, 'ACTION: noop')[1]
        trajectory = [observation]

        self.next_done = torch.zeros(1).to(self.device)
        self.next_obs = self.agent.tokenizer(observation, return_tensors="pt", padding='max_length', max_length = self.obs_length)["input_ids"].to(self.device)

        self.policy_optimizer.param_groups[0]["lr"] = frac * self.policy_learning_rate
        self.value_optimizer.param_groups[0]["lr"] = frac * self.value_learning_rate
        
        gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
        logger.info(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

        rewards = 0
        for step in range(0, self.num_steps):
            step_cnt += 1 
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done


            with torch.no_grad():
                next_obs_str = self.agent.tokenizer.decode(self.next_obs[0])
                curr_obs = self.prompt + trajectory[0]+self.user

                # gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
                # logger.info(r'after next_obs_str：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

                action, logprob, _, value = self.agent.get_action_and_value([curr_obs], self.actionlist)
                
                # from  numba import cuda
                # device = cuda.get_current_device()
                # device.reset()
                # cuda.select_device(0) 
                # cuda.close() 

                gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
                logger.info(r'after get_action_and_value {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
                action_str = 'ACTION: '+self.action_conv[self.actionlist[action.item()]]

                #gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
                # logger.info(r'after action_str ：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
                res, next_obs, reward, done, achievement = self.controller_steps(env, action_str)
  
                # gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
                # logger.info(r'after controller_steps：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

                trajectory.append('step '+str(step)+action_str+' '+'Observation: '+next_obs)
                trajectory = self.cutoff_obs(trajectory)
                #logger.info(str(trajectory))
                logger.info('###reward: {}, step: {}'.format(reward, step))
                self.values[step] = value.flatten()

            #logger.info('action:'+str(action))
            self.actions[step] = action
            self.logprobs[step] = logprob
            rewards += reward

            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1) 
            self.next_obs = self.agent.tokenizer(next_obs, return_tensors="pt", padding='max_length', max_length = self.obs_length)["input_ids"][:,:210].to(self.device)
            self.next_obs, next_done = torch.Tensor(self.next_obs).to(self.device), torch.Tensor([done]).to(self.device)
            self.steps[step] = torch.Tensor([1]).to(self.device)  

        # gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
        # logger.info(r'after step：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))
        # bootstrap value if not done
        with torch.no_grad():

            next_obs_str = self.agent.tokenizer.decode(self.next_obs[0])
            next_value = self.agent.get_value([next_obs_str]).reshape(1, -1)  
            advantages = torch.zeros_like(self.rewards).to(self.device)
            # lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]

                discount = torch.pow(self.gamma, self.steps[t])
                delta = self.rewards[t] + discount * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = delta 
                # logger.info(discount * nextvalues * nextnonterminal)
                # logger.info('nextvalues: '+str(nextvalues))
                # logger.info(nextnonterminal)
                # logger.info(delta)
            returns = advantages +self.values

        # logger.info('advantages:'+str(advantages))
        #logger.info('returns:'+str(returns))
        # flatten the batch
        b_obs = self.obs.reshape((-1,) + (self.obs_length,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) )
        #logger.info('self.actions  '+str(self.actions))

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        # logger.info('b_advantages:'+str(b_advantages))


        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        kl_explode = False
        policy_update_steps = 0
        pg_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        old_approx_kl = torch.tensor(0)
        approx_kl = torch.tensor(0)
        total_approx_kl = torch.tensor(0)
        
        # gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
        # logger.info(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

        for epoch in range(self.update_epochs):
            if kl_explode:
                break
            #update value
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.value_minibatch_size):
                end = start + self.value_minibatch_size
                mb_inds = b_inds[start:end][0]   ##extract str of index from array([index])

                # logger.info(self.obs)
                # logger.info(self.obs.reshape((-1,)).shape)
                # logger.info(b_obs.shape)
                # logger.info(b_inds)
                # logger.info(mb_inds)
                # logger.info(b_obs[mb_inds].int())

                b_obs_str = self.agent.tokenizer.decode(b_obs[mb_inds].int())
                newvalue = self.agent.get_value([b_obs_str])
                #logger.info('newvalue:'+str(newvalue))

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    # logger.info('b_returns:'+str(b_returns))
                    # logger.info('mb_inds:'+str(mb_inds))
                    # logger.info('v_loss_unclipped:'+str(v_loss_unclipped))
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = v_loss * self.vf_coef
                # logger.info('v_clipped:'+str(v_clipped))
                # logger.info('v_loss_clipped:'+str(v_loss_clipped))
                # logger.info('v_loss_max:'+str(v_loss_max))
                # logger.info('v_loss:'+str(v_loss))
                # logger.info('loss:'+str(v_loss))

                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
            
            logger.info('Update value')
            logger.info('v_loss  '+str(v_loss.item()))

            del v_loss, b_values
            torch.cuda.empty_cache()
            gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
            logger.info(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))

            self.policy_optimizer.zero_grad()            
            #update policy
            print('---------------------------------')
            for start in range(0, self.batch_size, self.policy_minibatch_size):

                if policy_update_steps % self.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + self.policy_minibatch_size
                # logger.info('start:'+str(start))
                # logger.info('end:'+str(end))
                # logger.info('b_inds[start:end]:'+str(b_inds[start:end]))
                # logger.info('b_actions:'+str(b_actions))
                # logger.info('b_actions[mb_inds]:'+str(b_actions[mb_inds]))
                mb_inds = b_inds[start:end][0]
                b_obs_str = self.agent.tokenizer.decode(b_obs[mb_inds].int())
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value([b_obs_str], self.actionlist, b_actions[mb_inds], is_warmup, return_value = False)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # logger.info('mb_inds:'+str(mb_inds))
                # logger.info('b_actions:'+str(b_actions))
                # logger.info('b_logprobs:'+str(b_logprobs))
                # logger.info('b_logprobs[mb_inds]:'+str(b_logprobs[mb_inds]))
                # logger.info('newlogprob:'+str(newlogprob))
                # logger.info('logratio:'+str(logratio))
                # logger.info('ratio:'+str(ratio))

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / self.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                #logger.info('mb_advantages:'+str(mb_advantages))

                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #logger.info('mb_advantages:'+str(mb_advantages))
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss
                loss /= self.gradient_checkpointing_steps
                
                loss.backward()
                
                # logger.info('policy_loss:'+str(pg_loss.item()))
                # logger.info('value_loss:'+str(v_loss.item()))
                # logger.info('mb_advantages:'+str(mb_advantages.item()))
                # logger.info('pg_loss1:'+str(pg_loss1.item()))
                # logger.info('pg_loss2:'+str(pg_loss2.item()))
                # logger.info('old_approx_kl:'+str(old_approx_kl.item()))
                # logger.info('approx_kl:'+str(approx_kl.item()))

                if policy_update_steps % self.gradient_checkpointing_steps == 0:
                    if self.target_kl is not None:
                        if total_approx_kl > self.target_kl:
                            self.policy_optimizer.zero_grad()
                            kl_explode = True
                            policy_update_steps -= self.gradient_checkpointing_steps
                            break                    
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()
                    self.policy_optimizer.zero_grad()    

                if start == self.batch_size -1:
                    pg_loss_tmp = pg_loss
                
                del newlogprob, logratio, ratio, entropy, entropy_loss, b_obs_str, loss, newvalue, pg_loss,  pg_loss1, pg_loss2, mb_advantages
                torch.cuda.empty_cache()
                del old_approx_kl, approx_kl # y_pred, y_true, var_y, explained_var
                torch.cuda.empty_cache()

        logger.info('Update policy')
        logger.info('pg_loss  '+str(pg_loss_tmp.item()))
        gpu_mem_total, gpu_mem_used, gpu_mem_free = self.get_gpu_mem_info(gpu_id=0)
        logger.info(r'当前显卡显存使用情况：总共 {} MB， 已经使用 {} MB， 剩余 {} MB'.format(gpu_mem_total, gpu_mem_used, gpu_mem_free))


        del b_logprobs, total_approx_kl, clipfracs, pg_loss_tmp
        gc.collect()
        torch.cuda.empty_cache()
        # logger.info('old_approx_kl  '+str(old_approx_kl.item()))
        # logger.info('approx_kl  '+str(approx_kl.item()))
        # logger.info('total_approx_kl  '+str(total_approx_kl.item()))
        logger.info('Finish epoch') 


        #self.agent.save(step_cnt // self.num_steps, "../result/")

        # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # if len(clipfracs) == 0:
        #     num_clipfracs = 0
        # else:
        #     num_clipfracs = np.mean(clipfracs)
    

        # writer.add_scalar("charts/policy_learning_rate", self.policy_optimizer.param_groups[0]["lr"], step_cnt)
        # writer.add_scalar("charts/value_learning_rate", self.value_optimizer.param_groups[0]["lr"], step_cnt)
        # writer.add_scalar("losses/value_loss", v_loss.item(), step_cnt)
        # writer.add_scalar("losses/policy_loss", pg_loss.item(), step_cnt)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), step_cnt)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), step_cnt)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), step_cnt)
        # writer.add_scalar("losses/total_approx_kl", total_approx_kl.item(), step_cnt)
        # writer.add_scalar("losses/policy_update_times", policy_update_steps // self.gradient_checkpointing_steps, step_cnt)
        # writer.add_scalar("losses/clipfrac", num_clipfracs, step_cnt)
        # writer.add_scalar("losses/explained_variance", explained_var, step_cnt)
        
        return step_cnt, rewards, achievement


        
    
        
    