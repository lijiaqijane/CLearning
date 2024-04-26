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
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from policy_llm import LLMAgent
from action import ReactAgent, logger


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_steps = 3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.policy_num_minibatches = self.num_steps
        self.value_num_minibatches = self.num_steps
        self.update_epochs = 1
        self.batch_size = self.num_steps
        self.policy_minibatch_size = int(self.batch_size // self.policy_num_minibatches)
        self.value_minibatch_size = int(self.batch_size // self.value_num_minibatches)
        self.total_timesteps = 500000
        self.seed = 1
        self.cuda = True
        self.policy_learning_rate = 5e-7
        self.value_learning_rate = 1e-5
        self.norm_adv = True
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.gradient_checkpointing_steps = 8
        self.resume =False
        self.load_path = "saved_models"

        self.normalization_mode = "word"
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
        
        self.policy_optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.policy_learning_rate, eps=1e-5, weight_decay=0)
        self.value_optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.agent.actor.parameters()), lr=self.value_learning_rate, eps=1e-5)
            
        # ALGO Logic: Storage setup
        self.obs_length = 200
        self.obs = torch.zeros((self.num_steps, 1)+ (self.obs_length,)).to(self.device) ##+ envs.single_observation_space.shape
        #logger.info('self.obs.shape: '+str(self.obs.shape))
        #logger.info('self.ob: '+str(self.obs))
        self.actions = torch.zeros((self.num_steps, 1)).to(self.device)  ##+ envs.single_action_space.shape
        #logger.info('self.actions.shape: '+ str(self.actions.shape))
        self.logprobs = torch.zeros((self.num_steps,1)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, 1)).to(self.device)
        self.dones = torch.zeros((self.num_steps, 1)).to(self.device)
        self.values = torch.zeros((self.num_steps, 1)).to(self.device)
        self.steps = torch.zeros((self.num_steps, 1)).to(self.device)
        self.action_list = ["Thought","Search","Ask"]

    def trainer(self, q, a, step_cnt, frac, is_warmup = False):

        reagent = ReactAgent(q, a)
        self.scratchpad = q
        self.next_done = torch.zeros(1).to(self.device)
        self.next_obs = self.agent.tokenizer(self.scratchpad, return_tensors="pt", padding='max_length', max_length = self.obs_length)["input_ids"].to(self.device)
        #logger.info('self.next_obs.shape: '+str(self.next_obs.shape))
        #logger.info('self.next_obs: '+str(self.next_obs))
        self.policy_optimizer.param_groups[0]["lr"] = frac * self.policy_learning_rate
        self.value_optimizer.param_groups[0]["lr"] = frac * self.value_learning_rate
            

        for step in range(0, self.num_steps):
            step_cnt += 1 
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done


            with torch.no_grad():
                next_obs_str = self.agent.tokenizer.decode(self.next_obs[0])
                #logger.info(next_obs_str)
                action, logprob, _, value = self.agent.get_action_and_value([next_obs_str],self.action_list)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # logger.info(self.actions.shape)
            # logger.info(value)
            # logger.info(logprob)

            action_str = self.action_list[action.item()]
            scratchpad, next_obs, reward, done = reagent.step(action_str, self.scratchpad)
            self.scratchpad += scratchpad

            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1) ##??
            self.next_obs = self.agent.tokenizer(next_obs, return_tensors="pt", padding='max_length', max_length = self.obs_length)["input_ids"].to(self.device)
            self.next_obs, next_done = torch.Tensor(self.next_obs).to(self.device), torch.Tensor([False]).to(self.device)
            self.steps[step] = torch.Tensor(action).to(self.device)  ##?? item['macro_action_steps'] for item in info


        # bootstrap value if not done
        with torch.no_grad():
            #logger.info(next_obs_str)
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
                advantages[t] = lastgaelam = delta 
                # logger.info(discount * nextvalues * nextnonterminal)
                # logger.info('nextvalues: '+str(nextvalues))
                # logger.info(nextnonterminal)
                # logger.info(delta)
            returns = advantages +self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + (self.obs_length,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

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

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
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

                self.value_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
            
            logger.info('Update value')

            self.policy_optimizer.zero_grad()            
            #update policy
            for start in range(0, self.batch_size, self.policy_minibatch_size):
                if policy_update_steps % self.gradient_checkpointing_steps == 0:
                    total_approx_kl = 0
                policy_update_steps += 1
                end = start + self.policy_minibatch_size
                mb_inds = b_inds[start:end][0]
                b_obs_str = self.agent.tokenizer.decode(b_obs[mb_inds].int())
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value([b_obs_str], self.action_list, b_actions.long()[mb_inds], is_warmup, return_value = False)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / self.gradient_checkpointing_steps
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]


                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss
                loss /= self.gradient_checkpointing_steps
                
                loss.backward()
                
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

            logger.info('Update policy')

        logger.info('Finish epoch')    
        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        self.agent.save(step_cnt + 1, f"./")

        return step_cnt

        
    
        
    