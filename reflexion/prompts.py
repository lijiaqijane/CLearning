from langchain.prompts import PromptTemplate

##Performing steps based solely on the previous scratchpad (thought/action/observation)
REACT_INSTRUCTION = REACT_INSTRUCTION = """You're a player trying to play the game of Crafter. 
Given the player’s current observation, you can only select one action in the list below:
Act: [Move West, 1], Act: [Move East, 2], Act: [Move North, 3], Act: [Move South, 4],
Act: [Do, 5], Act: [Sleep, 6], Act: [Place Stone, 7], Act: [Place Table, 8], Act: [Place Furnace, 9],
Act: [Place Plant, 10], Act: [Make Wood Pickaxe, 11], Act: [ Make Stone Pickaxe, 12], Act: [Make Iron Pickaxe, 13], 
Act: [Make Wood Sword, 14], Act: [Make Stone Sword, 15], Act: [Make Iron Sword, 16] 

Here are some examples:
{examples}
(END OF EXAMPLES)

Task: {task}
The player’s in game observation and previous experience: {get_observation}
"""


##Performing steps based on the previous scratchpad (thought/action/observation) and reflections
REACT_REFLECT_INSTRUCTION ="""You’re a player trying to play the game of Crafter. Solve the decison-making task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be two types: 
(1)Act [action to take, its serial number and numbers of action], which given the player’s current observation, you need to choose the next executable action to finish the task. Output the answer in this format: 'next action, its serial number, number of action'
(2)Finish [completed task name], which you have finished the task.

Here are the list of all the executable actions to take and its prerequisite:
1. Move West: Flat ground west of the agent.
2. Move East: Flat ground east of the agent.
3. Move North: Flat ground north of the agent.
4. Move South: Flat ground south of the agent.
5. Do: Facing creature or material; have necessary tool.
6. Sleep: Energy level is below maximum.
7. Place Stone: Stone in inventory.
8. Place Table: Wood in inventory.
9. Place Furnace: Stone in inventory.
10. Place Plant: Sapling in inventory.
11. Make Wood Pickaxe: Nearby table; wood in inventory.
12. Make Stone Pickaxe: Nearby table; wood, stone in inventory.
13. Make Iron Pickaxe: Nearby table, furnace; wood, coal, iron an inventory.
14. Make Wood Sword: Nearby table; wood in inventory.
15. Make Stone Sword: Nearby table; wood, stone in inventory.
16. Make Iron Sword: Nearby table, furnace; wood, coal, iron in inventory.
0. Noop: Always applicable.

Here are some examples:
{examples}
(END OF EXAMPLES)

{reflections}

Task: {task}
The player’s in game observation and previous experience for reference: {get_observation}"""

##Reflecting based on the previous scratchpad (thought/action/observation)
REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self-reflection. \
You will be given a previous reasoning trial in which you had access to an external database and a task to complete in Crafter. \
You were unsuccessful in completing the task either because you finish the current trial with Finish[<task name>] without completing the task, or you used up your set number of reasoning steps. \
In a few sentences, diagnose a possible reason for failure to complete the task and devise a new, concise, high-level plan that aims to mitigate the same failure. Use complete sentences.  
Here are some examples:
{examples}

Previous trial:{scratchpad}
Task: {task}

Reflection:"""

react_agent_prompt = PromptTemplate(
    input_variables=["examples", "task", "get_observation"],
    template=REACT_INSTRUCTION,
)

react_reflect_agent_prompt = PromptTemplate(
    input_variables=["examples", "reflections", "task", "get_observation"],
    template=REACT_REFLECT_INSTRUCTION,
)

reflect_prompt = PromptTemplate(
    input_variables=["examples", "task", "scratchpad"],
    template=REFLECT_INSTRUCTION,
)

REFLECTION_HEADER = 'You have attempted to complete the task before and failed. \
The following reflection(s) provide a plan to avoid failing to complete the task in the same way you did previously. \
Use them to improve your strategy of correctly completing the given task.\n'

REFLECTION_AFTER_LAST_TRIAL_HEADER = '\
The following reflection(s) provide a plan to avoid failing to complete the task in the same way you did previously. \
Use them to improve your strategy of correctly completing the given task.\n'

LAST_TRIAL_HEADER = 'You have attempted to complete the task before and failed. \
Below is the last trial you attempted to complete the task.\n'

FEEDBACK_INSTRUCTION = """There are two roles (Student and Teacher) in the decision-making task below. \
The Student is unsuccessful in solving the task below because it has limited knowledge and experience about the game of Crafter. \
You are the Teacher who is an expert in Crafter and can provide additional instructions about how to complete the task in one or two sentence as feedback for the Student. \
You will be given historical trajectories of the Student including its steps of reasoning, observations and actions taken under the condition. \
And you will also be given the achievement goal and the groundtruth of the task with its corresponding sequence of subgoals to finish. \

Here are some examples:
{examples}

Task: {task}

Subgoal sequence and the completed times each subgoal of the task:
{subgoals}

Student:
{scratchpad}

The current observation of Student:
{observation}

Teacher:
Feedback:"""

feedback_agent_prompt = PromptTemplate(
    input_variables=["examples", "task", "subgoals", "observation", "scratchpad"],
    template=FEEDBACK_INSTRUCTION,
)
