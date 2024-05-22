import re, string, os, json
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM, get_similarity_encoder, get_vectordb
from policy_llm import LLMAgent
from prompts import react_agent_prompt, feedback_agent_prompt
from fewshots import CRAFTER_SAMPLE,  FEEDBACKS
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.llms import OpenAI
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import logging
logger = logging.getLogger()
logger.setLevel('INFO')
formatter = logging.Formatter()
chlr = logging.StreamHandler() 
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  
fhlr = logging.FileHandler('Log1.log')
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)
import sys
log_print = open('Error1.log', 'w')
sys.stdout = log_print
sys.stderr = log_print

class ReactAgent:
    def __init__(self,
                 task,
                 env,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 feedback_prompt: PromptTemplate = feedback_agent_prompt,
                 interact_llm : AnyOpenAILLM = AnyOpenAILLM(),     
                 sim_encoder = get_similarity_encoder(),
                 retriever = get_vectordb()[0],
                 collection = get_vectordb()[1]
                 ) -> None:
        self.task = task
        self.agent_prompt = agent_prompt
        self.react_examples = CRAFTER_SAMPLE
        self.feedback_examples = FEEDBACKS
        self.interact_llm = interact_llm
        self.feedback_prompt = feedback_prompt
        self.Agent = LLMAgent()
        self.llm = LLMAgent().actor
        self.retriever = retriever
        self.collection = collection
        self.sim_encoder = sim_encoder
        self.previous_action = []
        self.previous_observation = []
        self.achieve_subgoal = []
        self.wrap_env = env

        
    def step(self, action, scratchpad, obs, rewards, action_list=['0.Noop']):
        
        self.wrap_env.previous_observation = obs  ##keep previous obs

        logger.info('------step-----'+action)
        action = action.split(':')
        action_type, action_content = action[0].strip(' '), action[1].strip(' ')
        scratchpad += action_type+': '+action_content

        # execute action = env.step
        if action_type == 'Search':
            docs = []
            context = ''
            if self.collection.count() > 0:
                try:
                    docs = self.retriever.get_relevant_documents(action_content)  
                    for i in range(len(docs)):
                        doc_details = docs[i].to_json()['kwargs']
                        self.title = doc_details['metadata']['title']
                        logger.info(str(i) + '【'+ self.title+ '】'+'\n')
                        context += doc_details['page_content']
                except Exception as e:
                    logger.info(e)
            argument = '$Retrieved context$: '+context

        elif action_type == 'Ask':
            feedback = self.get_feedback(obs, scratchpad)
            argument = '$Feedback$: '+ feedback

        elif action_type == 'Thought':
            argument = ''
        elif action_type == 'Act':
            try:
                executable_actions = self.wrap_env.get_executable_actions

                argument = action_content.split(', ')


                ## if act: 补全3个参数
                # if len(argument) != 3 or not all(
                #         items in executable_actions.items() for items in {int(argument[1]): argument[0]}.items()):
                #     #scratchpad += 'Invalid Act format. Vaild format is <action, number, times>.'
                #     pass
                # else:
                #     action_list = []
                #     for _ in range(int(argument[2])):
                #         action_list.append(argument[1] + '.' + argument[0])
                ## if act: 补全2个参数
                if len(argument) != 2 or not all(items in executable_actions.items() for items in {int(argument[1]): argument[0]}.items()):
                    pass
                else:
                    action_list = [argument[1] + '.' + argument[0]] 

            except Exception as e:
                #scratchpad += f'Could not execute this action, please try again.'
                pass
        else:
            argument = 'Invalid Action!!!!'
        #scratchpad += ' '+ argument
        logger.info('------argument---'+str(argument))

        logger.info('------action-----'+str(argument))
        step_res = self.wrap_env.steps(actions_list=action_list)
        obs = str(step_res[0])
        if obs not in self.wrap_env.previous_observation:
            scratchpad += ' Observation: '+obs
            logger.info('------obs-----'+' Observation: '+obs)

        finished = False
        if self.if_task_finished() > 0:
            rewards += self.if_task_finished()  #reward when task finish
            finished = True
        logger.info('##########rewards: '+str(rewards))
        rewards = rewards - 1  #reward at each time step


        achievement = self.wrap_env.achievements
        for key in achievement.keys():  #??any goal achieved is OK, not equals to the given task
            if self.wrap_env.achievements[key] != achievement[key]: ##前后ach不同表示有新增ach
                self.achieve_subgoal.append(key)
                self.previous_action.append(self.wrap_env.previous_action)
                self.previous_observation.append(self.wrap_env.previous_observation)
                rewards += 0.2 #reward when subgoal finish
                self.wrap_env.previous_action = []
                self.wrap_env.previous_observation = ''
                #break  ##如果加break就是检测到第一个新增subgoal就停止,否则检测所有新增

        return scratchpad, obs, rewards, finished, self.achieve_subgoal, self.previous_action, self.previous_observation

    def update_memory(self, subgoals, achieve_subgoal, pre_action, pre_observation):
        logger.info('--------------------------Mem upd----------------------------')
        ## 某个task的所有subgoal_sequence
        mem = 'The subgoal sequence for {} is '.format(self.task)
        for action in subgoals:
            mem += '{} {} times, '.format(str(action[0]), action[1])
        self.collection.add(
            documents=[mem],
            metadatas=[{"content": "task-subgoal"}],
            ids=["id_" + self.task])
        logger.info('***********mem1:' + mem)

        ## curr_obs + 某个subgoal_sequence
        if len(pre_action) == len(pre_observation) and len(pre_observation) == len(achieve_subgoal): ##什么情况下可能不等于
            for i in range(len(pre_action)):
                ach_subgoal = ''
                ach_subgoal += 'Based on the observation \'{}\', the sequence of actions to complete \'{}\' is '.format(
                    pre_observation[i], achieve_subgoal[i])
                for act in pre_action[i]:
                    ach_subgoal += '{}, '.format(act)

                self.collection.add(
                    documents=[ach_subgoal],
                    metadatas=[{"content": "subgoal-action"}],
                    ids=["id_" + achieve_subgoal[i] + pre_observation[i]])
                logger.info('***********mem2:' + ach_subgoal)

    def get_next_action(self, scratchpad, k_sent):
        action = self.prompt_agent(scratchpad, k_sent)
        return action

    def if_task_finished(self):  # how to get the progress if task finished
        return self.wrap_env.done
    
    def prompt_agent(self, scratchpad, k_sent) -> str:
        pmt= self._build_agent_prompt(scratchpad)
        oup = self.Agent.get_model(pmt, k_sent)
        return oup

    def _build_agent_prompt(self, scratchpad) -> str:
        return self.agent_prompt.format(
            examples=self.react_examples,
            task=self.task,
            get_observation= scratchpad)
    

    def get_feedback(self, observation, scratchpad) -> str:
        fd = format_step(self.interact_llm(self._build_feedback_prompt(observation, scratchpad)))
        sub = {}
        for s in self.wrap_env.subgoal:
            sub[s[0]] = self.wrap_env.achievements[s[0]]
        return fd

    def _build_feedback_prompt(self, observation, scratchpad) -> str:
        subgoals = ''
        if type(self.wrap_env.subgoal) is dict:
            for task in self.wrap_env.subgoal:
                subgoals += 'Task: {}'.format(task) + '\n'
                for goal in self.wrap_env.subgoal[task]:
                    subgoals += 'Subgoal: {}, the number of times Student need to complete: {}, the number of times Student have already completed: {}'.format(
                        goal[0], str(goal[1]), str(self.wrap_env.achievements[goal[0]]))
        else:
            for goal in self.wrap_env.subgoal:
                subgoals += 'Subgoal: {}, the number of times Student need to complete: {}, the number of times Student have already completed: {}'.format(
                    goal[0], str(goal[1]), str(self.wrap_env.achievements[goal[0]]))


        return self.feedback_prompt.format(
            examples=self.feedback_examples,
            task=self.task,
            subgoals=subgoals,
            scratchpad= scratchpad,
            observation=observation)

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None

def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n\n', ' ').replace('\n', ' ').replace('\'', '')


def format_action(step: str) -> str:
    step = step.split(':')[0].strip('Thought').strip('Search').strip('Feedback')
    return step.strip('\n').strip().replace('\n\n', ' ').replace('\n', ' ').replace('\'', '')



def embsim_match(answer, key, encoder, embsim_rate=0.9):
    embeddings = encoder.encode([answer, key])
    similarity = cosine_similarity(embeddings)[0][1]
    
    if similarity > embsim_rate:
        return True, similarity
    else:
        return False, similarity

def exempt_label(answer, key, encoder):
    cand = answer.replace(',','.').split('.')
    new_cad = [ s for s in cand if not embsim_match(s, key, encoder, 0.85)[0]]
    return ','.join(new_cad)

