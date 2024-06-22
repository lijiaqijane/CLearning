

from openai import AzureOpenAI
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils import *
from crafter.api.envWrapper import *
from crafter.api.controller import *
import argparse
import pathlib

import pygame
import crafter, os, shutil
import logging
from datetime import datetime
REGION = "northcentralus" 
MODEl = "gpt-4-0125-preview"

API_KEY = "19ac9a9a67631624f814a75c4140827a"

API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy" 
ENDPOINT = f"{API_BASE}/{REGION}"

llm = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
    )


def llm_invoke(system_message, human_message ) -> str:
    trial = 0
    while trial < 20:
        try:
            response = llm.chat.completions.create( 
                model=MODEl,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": human_message}
                ],
                temperature=0.7,
                stop='\n'
            )
        except:
            trial += 1
            continue
        else:
            break

    return response.choices[0].message.content


def render_system_message() -> str:
        
    system_template = load_prompt('/scratch/nlp/lijiaqi/CLearning/reflexion/Crafter/ReAct_cor/system_prompt.txt')
    
    programs = load_text('/scratch/nlp/lijiaqi/CLearning/reflexion/Crafter/ReAct_cor/programs.txt')
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(
                system_template
    )

    system_message = system_message_prompt.format(
        programs=programs
    )
    assert isinstance(system_message, SystemMessage)
    return system_message.content

def render_human_message(env_wrap:envWrapper) -> str:
    info = env_wrap._env.info
    try:
        result = ""
        result = describe_status(info)
        result += describe_env(info)
        result += describe_inventory(info)
        
        return result.strip()
    except:
        return "Error, you are out of the map."



# main function
def main():
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
    
    parser.add_argument('--gen_world', type=boolean, default=False)
    args = parser.parse_args()

    # pygame.init()
    # screen = pygame.display.set_mode(args.window)
    # clock = pygame.time.Clock()

    # record_path = args.load_world / args.model
    

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_filename = f"{timestamp}-{MODEl}"

    env = crafter.Env(seed= args.seed, args = args)
    #env = crafter.Recorder(env, record_path / log_filename)

    env.reset()

    #logger = setup_logger(record_path / log_filename / "logger.txt")

    obs, reward, done, info = env.step(0)
    

    bot = AgentController(env)
    env_wrap = envWrapper(env)

    done = False

    trajectories = []
    step = 0
    system_message = render_system_message()
    human_message = render_human_message(env_wrap)
    trajectories.append((step, human_message, ''))
    print(system_message)
    print(human_message)
    
    while not done:
        response = llm_invoke(system_message, human_message)
        print(response)
        if response.startswith('ACTION: '):
            action = response.split(': ')[1]
            try:
                if '(' not in action:
                    action = f"{action}()"
                
                exec(f"bot.{action}")
            except:
                observation = "The generated action is not valid. Please check the available actions."
            else:
                observation = render_human_message(env_wrap)
            
            done = env.info['done']

        elif response.startswith('THINK: '):
            observation = "OK."
            # trajectories.append((step, response, observation))

        else:
            observation ="The output is not recognized."
        
        step += 1
        trajectories.append((step, response, observation))
        human_message = "".join([f"{c}\n{d}\n" for i, c, d in trajectories[-10:]])
        print(observation)
        if done:
            break
        
    pygame.quit()
    

if __name__ == "__main__":
    main()

