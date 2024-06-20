# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import gym
import torch.optim as optim

from .llama3_formatter import ChatFormat, Dialog, Message
from .crafter import crafter_env
from .policy_llm import LLMAgent
from .action import ReactAgent, logger


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    ALLOW_GOAL = True
    ALLOW_TRAJ = True
    ALLOW_STEP = True
    ALLOW_SUMMARY = True
    ALLOW_RECORD = True
    ALLOW_REFLECT = True
    ALLOW_THINK = True

    def __init__(self, max_steps, max_obs):
        super().__init__()

        self.num_steps = max_steps
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.policy_num_minibatches = self.num_steps
        self.value_num_minibatches = self.num_steps
        self.update_epochs = 1
        self.batch_size = self.num_steps
        self.policy_minibatch_size = int(self.batch_size // self.policy_num_minibatches)
        self.value_minibatch_size = int(
            self.batch_size // (self.value_num_minibatches * 0.25)
        )
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
        self.resume = False
        self.load_path = f"{os.path.dirname(__file__)}/result"
        self.llama_formatter = ChatFormat()
        self.normalization_mode = "word"
        self.system_prompt = Message(
            role="system",
            content="""You are currently in the game "Crafter" through textual APIs. In each turn, you must create an API call message based on the textual observation of what you see and your current status. Your actions will result in rewards or punishments, and your goal is to maximize the total reward.
        
The observation will follow the format:
```
You see {a list of things you can see}

You face {the thing close to you}

You status: {the detailed status of you}

You have {your inventory}
```

Your output message MUST strictly adhere to the format:

```
Act: {instruction}
```
It is important that all possible instructions are list below:
    ["Noop", "Move West", "Move East", "Move South", "Move North", "Do", "Sleep", "Place Stone", "Place Table", "Place Furnace", "Place Plant", "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe", "Make Wood Sword", "Make Stone Sword", "Make Iron Sword"]
""",
        )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.cuda else "cpu"
        )

        if self.resume:
            self.agent = LLMAgent(
                max_obs=max_obs,
                normalization_mode=self.normalization_mode,
                load_path=self.load_path,
                load_8bit=True,
            )
        else:
            self.agent = LLMAgent(
                max_obs=max_obs,
                normalization_mode=self.normalization_mode,
                load_8bit=True,
            )

        # logger.info(list(self.agent.actor.parameters()))
        # logger.info(list(filter(lambda p: p.requires_grad, self.agent.actor.parameters())))
        self.policy_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.agent.actor.parameters()),
            lr=self.policy_learning_rate,
            eps=1e-5,
            weight_decay=0,
        )
        self.value_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.agent.critic.parameters()),
            lr=self.value_learning_rate,
            eps=1e-5,
        )

        # ALGO Logic: Storage setup
        self.obs_length = max_obs
        self.obs = torch.zeros((self.num_steps, 1) + (self.obs_length,)).to(
            self.device
        )  ##+ envs.single_observation_space.shape
        self.actions = torch.zeros((self.num_steps, 1)).to(
            self.device
        )  ##+ envs.single_action_space.shape
        self.logprobs = torch.zeros((self.num_steps, 1)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, 1)).to(self.device)
        self.dones = torch.zeros((self.num_steps, 1)).to(self.device)
        self.values = torch.zeros((self.num_steps, 1)).to(self.device)
        self.steps = torch.zeros((self.num_steps, 1)).to(self.device)
        # self.action_list = ["Thought","Search","Ask"]
        self.candidate_action_num = 1

    @DeprecationWarning
    def format_action(self, action_list):
        actions = []
        for i in action_list:
            if (
                i == ""
                or i == " "
                or (
                    ("Thought: " not in i)
                    and ("Ask: " not in i)
                    and ("Search: " not in i)
                )
            ):
                i = "Thought: " + i
            actions.append(i)
        return actions

    def cutoff_traj(self, traj):
        if len(traj) > 3:
            traj.pop(0)
        return traj

    # !!! this prompts = obs, don't add thought here !!!
    def pack_prompts(self, traj, obs, env_step) -> str:
        prefix = []
        suffix = []
        if self.ALLOW_GOAL:
            # TODO update task list during running?
            task_list = [
                "place_plant",
                "collect_wood",
                "place_table",
                "make_wood_sword",
                "make_wood_pickaxe",
                "eat_plant",
                "collect_coal",
                "collect_stone",
                "place_stone",
                "place_furnace",
                "make_stone_sword",
                "make_stone_pickaxe",
                "collect_iron",
                "make_iron_sword",
                "make_iron_pickaxe",
                "collect_diamond",
                "collect_drink",
                "collect_sapling",
                "defeat_skeleton",
                "defeat_zombie",
                "eat_cow",
                "wake_up",
            ]
            task_str = ",".join(task_list)
            prefix.append(
                f"""Your goal is to finish the below tasks:\n```[{task_str}]```"""
            )

        if self.ALLOW_TRAJ:
            traj_str = "\n".join(traj)
            prefix.append(f"""The recent history of your actions:\n```{traj_str}```""")

        # if self.ALLOW_REFLECT:
        #     # TODO you got a positive/negative reward ...
        #     pass

        if self.ALLOW_RECORD:
            # TODO you have finished ...
            pass

        if self.ALLOW_STEP:
            prefix.append(f"You have acted {env_step}/500 steps")

        prompt_content = obs
        if len(prefix) > 0:
            prompt_content = "\n".join(prefix) + "\n" + prompt_content
        if len(suffix) > 0:
            prompt_content = prompt_content + "\n" + "\n".join(suffix)

        full_user_prompt = Message(role="user", content=prompt_content)
        # obs_user_prompt = Message(role="user", content=obs)

        return self.llama_formatter.format_dialog_prompt(
            [
                self.system_prompt,
                full_user_prompt,
            ]
        )

    def think(self):
        # TODO
        pass

    def trainer(
        self, task, step_cnt, frac, writer, is_warmup=False
    ):  ##？？做对提前结束
        self.next_done = torch.zeros(1).to(self.device)

        self.policy_optimizer.param_groups[0]["lr"] = frac * self.policy_learning_rate
        self.value_optimizer.param_groups[0]["lr"] = frac * self.value_learning_rate

        # prepare craft env
        env = gym.make("smartplay:Crafter-v0")
        env = crafter_env.WrapEnv(env)
        env.set_task(task)
        next_obs_str = env.reset()
        reagent = ReactAgent(task, env, self.agent)
        rewards = 0
        trajectory = []

        for step in range(0, self.num_steps):
            step_cnt += 1

            with torch.no_grad():

                action_list = [
                    "Action: Noop",
                    "Action: Move West",
                    "Action: Move East",
                    "Action: Move North",
                    "Action: Move South",
                    "Action: Do",
                    "Action: Sleep",
                    "Action: Place Stone",
                    "Action: Place Table",
                    "Action: Place Furnace",
                    "Action: Place Plant",
                    "Action: Make Wood Pickaxe",
                    "Action: Make Stone Pickaxe",
                    "Action: Make Iron Pickaxe",
                    "Action: Make Wood Sword",
                    "Action: Make Stone Sword",
                    "Action: Make Iron Sword",
                ]

                prompt_str = self.pack_prompts(
                    trajectory, next_obs_str, reagent.step_count
                )

                # concat prompt + action
                # logger.info(self.prompt + '\n'.join(trajectory))
                action, logprob, _, value, encoded_prompt = (
                    self.agent.get_action_and_value(prompt_str, action_list)
                )

                self.next_obs = encoded_prompt
                self.values[step] = value.flatten()
                # logger.info('logprob  '+str(logprob))

            # store step
            self.obs[step] = self.next_obs
            self.dones[step] = self.next_done
            self.actions[step] = action
            self.logprobs[step] = logprob

            ##orgin
            # action_str = action_list[action.item()]

            executable_actions = env.get_executable_actions()
            executable_action = action_list[action.item()]
            action_str = (
                executable_action
                + ", "
                + str(executable_actions[executable_action.split(": ")[1]])
            )
            logger.info("get_action_and_value:" + str(action_str) + "   " + str(value))
            (
                trajectory,
                next_obs_str,
                reward,
                achievement,
                done,
                ach_subg,
                preact,
                preobs,
            ) = reagent.step(action_str, trajectory, next_obs_str)

            trajectory = self.cutoff_traj(trajectory)

            logger.info("###########reward: {}, step: {}".format(reward, step))
            rewards += reward

            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)  ##??
            next_done = torch.Tensor([done]).to(self.device)
            self.steps[step] = torch.Tensor(action).to(
                self.device
            )  ##?? item['macro_action_steps'] for item in info

        # memory update
        # reagent.update_memory(env.subgoal, ach_subg, preact, preobs)

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
                delta = (
                    self.rewards[t]
                    + discount * nextvalues * nextnonterminal
                    - self.values[t]
                )
                advantages[t] = delta
                # logger.info(discount * nextvalues * nextnonterminal)
                # logger.info('nextvalues: '+str(nextvalues))
                # logger.info(nextnonterminal)
                # logger.info(delta)
            returns = advantages + self.values

        # logger.info('advantages:'+str(advantages))
        # logger.info('returns:'+str(returns))
        # flatten the batch
        b_obs = self.obs.reshape((-1,) + (self.obs_length,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,))
        # logger.info('self.actions  '+str(self.actions))

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

        torch.cuda.empty_cache()

        for epoch in range(self.update_epochs):
            if kl_explode:
                break
            # update value
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.value_minibatch_size):
                end = start + self.value_minibatch_size
                mb_inds = b_inds[start:end][
                    0
                ]  ##extract str of index from array([index])

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

            logger.info("Update value")
            logger.info("value_loss  " + str(loss.item()))
            logger.info("v_loss  " + str(v_loss.item()))

            self.policy_optimizer.zero_grad()
            # update policy
            print("---------------------------------")
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
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs_str,
                    action_list,
                    b_actions[mb_inds],
                    is_warmup,
                    return_value_and_info=False,
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # logger.info('mb_inds:'+str(mb_inds))
                # logger.info('b_actions:'+str(b_actions))
                # logger.info('b_logprobs:'+str(b_logprobs))
                # logger.info('b_logprobs[mb_inds]:'+str(b_logprobs[mb_inds]))
                # logger.info('newlogprob:'+str(newlogprob))
                # logger.info('logratio:'+str(logratio))
                # logger.info('ratio:'+str(ratio))

                # logratio = newlogprob.item() - b_logprobs[mb_inds].item()
                # ratio = logratio.exp()
                # logger.info('logratio:'+str(logratio))
                # logger.info('ratio:'+str(ratio))

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    total_approx_kl += approx_kl / self.gradient_checkpointing_steps
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                # logger.info('mb_advantages:'+str(mb_advantages))

                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # logger.info('mb_advantages:'+str(mb_advantages))
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
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
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.max_grad_norm
                    )
                    self.policy_optimizer.step()
                    self.policy_optimizer.zero_grad()

            logger.info("Update policy")
            logger.info("policy_loss  " + str(loss.item()))
            logger.info("pg_loss  " + str(pg_loss.item()))

        logger.info("old_approx_kl  " + str(old_approx_kl.item()))
        logger.info("approx_kl  " + str(approx_kl.item()))
        logger.info("total_approx_kl  " + str(total_approx_kl.item()))
        logger.info("Finish epoch")
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if len(clipfracs) == 0:
            num_clipfracs = 0
        else:
            num_clipfracs = np.mean(clipfracs)

        writer.add_scalar(
            "charts/policy_learning_rate",
            self.policy_optimizer.param_groups[0]["lr"],
            step_cnt,
        )
        writer.add_scalar(
            "charts/value_learning_rate",
            self.value_optimizer.param_groups[0]["lr"],
            step_cnt,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), step_cnt)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), step_cnt)
        writer.add_scalar("losses/entropy", entropy_loss.item(), step_cnt)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), step_cnt)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), step_cnt)
        writer.add_scalar("losses/total_approx_kl", total_approx_kl.item(), step_cnt)
        writer.add_scalar(
            "losses/policy_update_times",
            policy_update_steps // self.gradient_checkpointing_steps,
            step_cnt,
        )
        writer.add_scalar("losses/clipfrac", num_clipfracs, step_cnt)
        writer.add_scalar("losses/explained_variance", explained_var, step_cnt)

        return step_cnt, rewards, achievement
