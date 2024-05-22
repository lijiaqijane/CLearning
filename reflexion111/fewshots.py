CRAFTER_SAMPLE = """
Task: Place Table.
Search: experience of the task Place Table.
Thought: I need to collect wood first to place table, then I need to collect wood to make wood pickaxe, finally I need to collect coal using the wood pickaxe.
Act: [Move South, 4]
Observation: There is water 7 steps to your south-east, grass 1 steps to your west, tree 2 steps to your south. You face grass at your front. You have nothing in your inventory.
Thought: I need to go to the front of the tree to get the wood.
Ask: How can i move to the front of the tree?
Observation: There is grass 1 steps to your west, tree 1 steps to your south. You face tree at your front. You have nothing in your inventory.
Thought: I need to collect wood now.
Act: [Do, 5]
Observation: There is grass 1 steps to your west, tree 3 steps to your west. You face grass at your front. Your inventory is 1 wood.
Ask: What else do i need to place table after i already have wood in my inventory?
Thought: I need to move to the tree in my south-west and collect wood.
Act: [Move West, 1]
Observation: There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 1 steps to your west. You face tree at your front. Your inventory is 1 wood.
Act: [Do, 5]
Observation: There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 3 steps to your south-east. You face grass at your front. Your inventory is 2 wood.
Search: Subgoal sequences of the task Place Table.
Thought: I have enough wood and can place table now.
Act: [Place Table, 8]
Observation:There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 3 steps to your south-east, table 1 step to your west. You face table at your front. You have nothing in your inventory.
Thought: I have already placed table.


Task: Make Wood Pickaxe.
Thought: I need to collect one wood to make wood pickaxe.
Act: [Move South, 4]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west, tree 1 steps to your south. You face tree at your front. You have nothing in your inventory.
Act: [Do, 5]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face grass at your front. Your inventory is 1 wood.
Search: Subgoal sequences of the task Make Wood Pickaxe.
Thought: I need make wood pickaxe near the table with a wood.
Act: [Move North, 3]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face table at your front. Your inventory is 1 wood.
Ask: What's the prerequisite of making wood pickaxe?
Thought: I can make wood pickaxe now.
Act: [Make Wood Pickaxe, 11]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face grass at your front. Your inventory is 1 wood pickaxe.
"""


FEEDBACKS = """
"""
