import re, string, os, json
from typing import List, Union, Literal
from enum import Enum
import tiktoken
from langchain.prompts import PromptTemplate
from llm import AnyOpenAILLM, get_similarity_encoder, get_vectordb
from policy_llm import LLMAgent
from prompts import reflect_prompt, react_agent_prompt, feedback_agent_prompt, react_reflect_agent_prompt, memupdate_agent_prompt
from prompts import REFLECTION_HEADER, LAST_TRIAL_HEADER, REFLECTION_AFTER_LAST_TRIAL_HEADER
from fewshots import WEBTHINK_SIMPLE6, REFLECTIONS, FEEDBACKS, UPDATES
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains.question_answering import load_qa_chain, LLMChain
import openai
import os
from langchain_community.llms import OpenAI
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import subprocess
from time import sleep
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain import LLMChain
from openai import AzureOpenAI
import logging
logger = logging.getLogger()
logger.setLevel('INFO')
formatter = logging.Formatter()
chlr = logging.StreamHandler() 
chlr.setFormatter(formatter)
chlr.setLevel('INFO')  
fhlr = logging.FileHandler('Error.log')
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)
import sys
log_print = open('Defalust.log', 'w')
sys.stdout = log_print
sys.stderr = log_print

class ReactAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 feedback_prompt: PromptTemplate = feedback_agent_prompt,
                 interact_llm : AnyOpenAILLM = AnyOpenAILLM(
                                            temperature=0,
                                            max_tokens=500,
                                            model_name="gpt-4",
                                            model_kwargs={"stop": "\n"},
                                            openai_api_key=os.environ['OPENAI_API_KEY']),     
                Agent  : LLMAgent = LLMAgent(),
                sim_encoder = get_similarity_encoder(),
                retriever = get_vectordb()[0],
                collection = get_vectordb()[1]
                 ) -> None:
        self.question = question
        self.answer = ''
        self.key = key
        self.agent_prompt = agent_prompt
        self.react_examples = WEBTHINK_SIMPLE6
        self.feedback_examples = FEEDBACKS
        self.interact_llm = interact_llm
        self.feedback_prompt = feedback_prompt
        self.llm = Agent.actor
        self.Agent = Agent
        self.retriever = retriever
        self.collection = collection
        self.sim_encoder = sim_encoder

     
    def step(self, action_type, scratchpad) -> None:
        
        logger.info('------action_type-----'+action_type)
        action_content = format_action(self.prompt_agent(scratchpad, action_type+': '))
        logger.info('------action_content-----'+action_content)
        scratchpad += ' '+action_type +': '+action_content

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
                    scratchpad += f'Could not find that page, please try again.'
            argument = '$Retrieved context$: '+context

        elif action_type == 'Ask':
            feedback = self.get_feedback(scratchpad)
            argument = '$Feedback$: '+ feedback

        elif action_type == 'Thought':
            argument = ''
        else:
            argument = 'Invalid Action!!!!'
        scratchpad += ' '+ argument
        logger.info('------argument-----'+argument)

        ## get next obs
        pmt = scratchpad+'Please generate the observation for question based on the given context.'
        obs = format_step(self.Agent.get_model(pmt)).strip(pmt)
        scratchpad += ' Observation: '+obs
        logger.info('------obs-----'+obs)
        

        ##get reward & done
        self.answer = obs
        signal = self.gpt_correct() 
        if signal:
            scratchpad += ' Answer is CORRECT'
            reward = 1
            finished = 1
        else:
            scratchpad += ' Answer is WRONG'
            reward = -1
            finished = 0
            
        return scratchpad, obs, reward, finished


    def prompt_agent(self, scratchpad, action_type) -> str:
        pmt= self._build_agent_prompt(scratchpad, action_type)
        oup = format_step(self.Agent.get_model(pmt).split(pmt)[1])
        return oup
    
    def _build_agent_prompt(self, scratchpad, action_type) -> str:
        return self.agent_prompt.format(
                            examples = self.react_examples,
                            question = self.question,
                            scratchpad = scratchpad,
                            action_type = action_type)

    def gpt_correct(self):
        Q, A, P = self.question, self.key, self.answer 
        #script_path = '/scratch/nlp/lijiaqi/curl.sh'
        args = "Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. \
    Please output True or False only. If there are expressions like 'I don't know' or 'I cant find' or 'The text doesn't provide' or 'it is not provided in the text', \
    then output 'None'."+"Question: "+Q+'  '+"groudtruth = " + A + '  '+ "predict_answer = "+P
        args = args.replace('"','^')
        while True:
            if '\n' in args:
                args = args.strip().replace('\n',' ')
            else:
                break
        ENDPOINT = f"https://api.tonggpt.mybigai.ac.cn/proxy/canadaeast"
        client = AzureOpenAI(
                    api_key="",
                    api_version="2024-02-01",
                    azure_endpoint=ENDPOINT,
                    )

        response = client.chat.completions.create(
            model="gpt-35-turbo-0125",
            messages=[
                {"role": "user", "content": args}
            ],
        )
        return response.choices[0].message.content


    def get_feedback(self, scratchpad) -> str:
        fdo = format_step(self.interact_llm(self._build_feedback_prompt(scratchpad)))
        fd = exempt_label(fdo, self.key, self.sim_encoder)
        logger.info('***********feedback:'+ fd)
        return fd

    def _build_feedback_prompt(self, scratchpad) -> str:
        return self.feedback_prompt.format(
                            examples = self.feedback_examples,
                            question = self.question,
                            scratchpad = scratchpad,
                            groundtruth = self.key)


### String Stuff ###
gpt2_enc = tiktoken.encoding_for_model("text-davinci-003")

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


