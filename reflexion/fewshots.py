CRAFTER_SAMPLE = """
Task: Place Table.
Search: experience of the task Place Table.
Act: Move South, 4, 3
Observation: There is water 7 steps to your south-east, grass 1 steps to your west, tree 2 steps to your south. You face grass at your front. You have nothing in your inventory.
Ask: How can i move to the front of the tree?
Observation: There is grass 1 steps to your west, tree 1 steps to your south. You face tree at your front. You have nothing in your inventory.
Act: Do, 5, 1
Observation: There is grass 1 steps to your west, tree 3 steps to your west. You face grass at your front. Your inventory is 1 wood.
Ask: What else do i need to place table after i already have wood in my inventory?
Act: Move West, 1, 2
Observation: There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 1 steps to your west. You face tree at your front. Your inventory is 1 wood.
Act: Do, 5, 1
Observation: There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 3 steps to your south-east. You face grass at your front. Your inventory is 2 wood.
Search: Subgoal sequences of the task Place Table.
Act: Place Table, 8, 1
Observation:There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 3 steps to your south-east, table 1 step to your west. You face table at your front. You have nothing in your inventory.


Task: Make Wood Pickaxe.
Act: Move South, 4, 1
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west, tree 1 steps to your south. You face tree at your front. You have nothing in your inventory.
Act: Do, 5, 1
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face grass at your front. Your inventory is 1 wood.
Search: Subgoal sequences of the task Make Wood Pickaxe.
Act: Move North, 3, 1
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face table at your front. Your inventory is 1 wood.
Ask: What's the prerequisite of making wood pickaxe?
Act: Make Wood Pickaxe, 11, 1
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face grass at your front. Your inventory is 1 wood pickaxe.

"""


FEEDBACKS = """
"""
