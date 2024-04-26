#基于previous scratchpad（thought/action/obs）+  reflections 进行step
from langchain.prompts import PromptTemplate

REACT_INSTRUCTION = """Solve a math QA task with interleaving action type: Thought, Search, Ask steps. \
Thought can reason about the current situation. \
Search searches and returns the relevant articles or paragraphs from external database as context. \
Ask can ask for help to an external expertise for additional knowledge and hints on the task.
You will be given the action type, please generate the action content. 

Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}{scratchpad}

{action_type}"""


REACT_REFLECT_INSTRUCTION = """Solve a math theorem proving task with interleaving Thought, Action, Observation steps. \
Thought can reason about the current situation, and Action can be two types: 
(1) Search[key words or phrases], which searches and returns the relevant premises or theorems from external database to apply.
(2) Finish[proof for given statement], which returns the complete proof and finishes the task.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Question: {question}{scratchpad}"""

##基于previous scratchpad（thought/action/obs） 进行reflect
REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. \
You will be given a previous reasoning trial in which you were given access to an external database and a math statement to prove. \
You were unsuccessful in completing the task either because you returned the wrong proof with Finish[<proof>], or you used up your set number of reasoning steps. \
In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:
Question: {question}{scratchpad}

Reflection:"""

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad","action_type"],
                        template = REACT_INSTRUCTION,
                        )

react_reflect_agent_prompt = PromptTemplate(
                        input_variables=["examples", "reflections", "question", "scratchpad"],
                        template = REACT_REFLECT_INSTRUCTION,
                        )

reflect_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REFLECT_INSTRUCTION,
                        )



REFLECTION_HEADER = 'You have attempted to finish the following task before and failed. \
The following reflection(s) give a plan to avoid failure in the same way you did previously. \
Use them to improve your strategy of correctly finish the given task.\n'

REFLECTION_AFTER_LAST_TRIAL_HEADER = '\
The following reflection(s) give a plan to avoid failure in the same way you did previously. \
Use them to improve your strategy of correctly finish the given task.\n'

LAST_TRIAL_HEADER = 'You have attempted to finish the following task before and failed. \
Below is the last trial you attempted to complete the task.\n'


FEEDBACK_INSTRUCTION = """There are two roles (Student and Teacher) in the math theorem proving task below. \
Student is unsuccessful in completing the task because it has limited relevant knowledge.\
You are the Teacher who is an expertise with rich knowledge in math theorem proving and can provide additonal facts in one or two sentence as feedback for Student. \
You will be given reasoning steps of Student in previous trajectory and the  "Groundtruth" which is direct answer. \
You will be punished if the feedback is semantically similar to "Groundtruth" or contains the same knowledge as "Groundtruth" in different expressions.\
Here are some examples:
{examples}

Question: {question}
Groundtruth: {groundtruth}

Student:
{scratchpad}

Teacher:
Feedback:"""

feedback_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad","groundtruth"],
                        template = FEEDBACK_INSTRUCTION,
                        )


MEMORY_UPDATE = """Given the latest relevant fact, please update/edit the existing memory based on the fact and then summary.\
If the given the fact has nothing to do with the existing memory and there is no need to update/edit, then output 'None'.
Here are some examples:
{examples}

Existing memory: 
{existing_memory}
Latest relevant fact: 
{acquired_fact}
Update and summarization: 
"""

memupdate_agent_prompt = PromptTemplate(
                        input_variables=["examples", "existing_memory","acquired_fact"],
                        template = MEMORY_UPDATE,
                        )