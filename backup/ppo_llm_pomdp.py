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
from reflexion.crafter import crafter_env
import torch.optim as optim
from torch.distributions.categorical import Categorical
from policy_llm import LLMAgent
from action import ReactAgent, logger
from torch.utils.tensorboard import SummaryWriter

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):

    def __init__(self, max_obs):
        super().__init__()

        self.num_steps = 2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.policy_num_minibatches = self.num_steps
        self.value_num_minibatches = self.num_steps
        self.update_epochs = 1
        self.batch_size = self.num_steps
        self.policy_minibatch_size = int(self.batch_size // self.policy_num_minibatches)
        self.value_minibatch_size = int(self.batch_size // self.value_num_minibatches)
        self.seed = 1
        self.cuda = True
        self.policy_learning_rate = 5e-7
        self.value_learning_rate = 1e-5
        self.norm_adv = False
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.target_kl = None
        self.gradient_checkpointing_steps = 8
        self.resume = False
        self.load_path = f"__path__}/result"
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
        #self.action_list = ["Thought","Search","Ask"]
        self.candidate_action_num = 1
        self.scratchpad = ''

    def format_action(self, action_list):
        actions = []
        for i in action_list:
            if i =='' or i ==' ' or (('Thought: ' not in i) and ('Ask: ' not in i) and ('Search: ' not in i)):
                i = 'Thought: '+i
            actions.append(i)
        return actions

    def cutoff_obs(self, traj):
        if len(traj) > 3:
            traj.pop(0)
        return traj

    def trainer(self, task, step_cnt, frac,  writer, is_warmup = False):  ##？？做对提前结束
        #logger.info('max_obs:'+str(self.obs_length))

        #prepare craft env
        env = gym.make("smartplay:Crafter-v0")
        env = crafter_env.WrapEnv(env)
        env.set_task(task)

        reagent = ReactAgent(task, env)  
        observation = str(env.steps(['0.Noop'])[0])  
        trajectory = [observation+'\n' ]
        self.scratchpad = '\n'.join(trajectory)

        self.next_done = torch.zeros(1).to(self.device)
        self.next_obs = self.agent.tokenizer(observation, return_tensors="pt", padding='max_length', max_length = self.obs_length)["input_ids"].to(self.device)

        self.policy_optimizer.param_groups[0]["lr"] = frac * self.policy_learning_rate
        self.value_optimizer.param_groups[0]["lr"] = frac * self.value_learning_rate
            
        for step in range(0, self.num_steps):
            step_cnt += 1 
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done


            with torch.no_grad():
                next_obs_str = self.agent.tokenizer.decode(self.next_obs[0])
                action_list = reagent.get_next_action(trajectory, self.candidate_action_num)
                
                # action_list = reagent.get_next_action(trajectory, self.candidate_action_num)
                
                # logger.info('action_list:'+str(action_list))
                # ##exp: random select from 4 action candidates
                # p = np.array([0.1,0.1,0.1,0.7])
                # org_action_list = action_list
                # action_list = [ org_action_list[np.random.choice([0,1,2,3],p=p.ravel())] ]
                # ##exp: random select from 4 action candidates
                action_list = ["Act: Move West, 1", "Act: Move East, 2", "Act: Move North, 3", "Act: Move South, 4",
"Act: Do, 5", "Act: Sleep, 6", "Act: Place Stone, 7", "Act: Place Table, 8", "Act: Place Furnace, 9",
"Act: Place Plant, 10", "Act: Make Wood Pickaxe, 11", "Act: Make Stone Pickaxe, 12", "Act: Make Iron Pickaxe, 13", 
"Act: Make Wood Sword, 14", "Act: Make Stone Sword, 15", "Act: Make Iron Sword, 16"]

                action, logprob, _, value = self.agent.get_action_and_value([next_obs_str], action_list)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # logger.info(self.actions.shape)
            # logger.info(value)
            # logger.info(logprob)

            action_str = action_list[action.item()]
            trajectory, next_obs, reward, achievement, done, ach_subg, preact, preobs = reagent.step(action_str,  trajectory , next_obs_str)
            trajectory = self.cutoff_obs(trajectory)
            logger.info('###########rewards: {}, step: {}'.format(reward, step))

            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1) ##??
            self.next_obs = self.agent.tokenizer(next_obs, return_tensors="pt", padding='max_length', max_length = self.obs_length)["input_ids"].to(self.device)
            self.next_obs, next_done = torch.Tensor(self.next_obs).to(self.device), torch.Tensor([done]).to(self.device)
            self.steps[step] = torch.Tensor(action).to(self.device)  ##?? item['macro_action_steps'] for item in info


        # memory update
        #reagent.update_memory(env.subgoal, ach_subg, preact, preobs)

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
        # logger.info('returns:'+str(returns))
        # flatten the batch
        b_obs = self.obs.reshape((-1,) + (self.obs_length,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) )
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
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value([b_obs_str], action_list, b_actions[mb_inds], is_warmup, return_value = False)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                # logger.info('mb_inds:'+str(mb_inds))
                # logger.info('b_actions:'+str(b_actions))
                # logger.info('b_logprobs:'+str(b_logprobs))
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

            logger.info('Update policy')

        logger.info('Finish epoch') 
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

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
        
        return step_cnt, reward, achievement

        
    
        
    
