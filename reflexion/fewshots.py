CRAFTER_SAMPLE = """
ACTION: [Move South, 4]
Observation: There is water 7 steps to your south-east, grass 1 steps to your west, tree 2 steps to your south. You face grass at your front. You have nothing in your inventory.
ACTION: [Do, 5]
Observation: There is grass 1 steps to your west, tree 3 steps to your west. You face grass at your front. Your inventory is 1 wood.
ACTION: [Move West, 1]
Observation: There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 1 steps to your west. You face tree at your front. Your inventory is 1 wood.
ACTION: [Do, 5]
Observation: There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 3 steps to your south-east. You face grass at your front. Your inventory is 2 wood.
ACTION: [Place Table, 8]
Observation:There is grass 2 steps to your north-east, sand 6 steps to your south-west, tree 3 steps to your south-east, table 1 step to your west. You face table at your front. You have nothing in your inventory.

Task: Make Wood Pickaxe.
ACTION: [Move South, 4]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west, tree 1 steps to your south. You face tree at your front. You have nothing in your inventory.
ACTION: [Do, 5]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face grass at your front. Your inventory is 1 wood.
ACTION: [Move North, 3]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face table at your front. Your inventory is 1 wood.
ACTION: [Make Wood Pickaxe, 11]
Observation: There is water 4 steps to your south-west, grass 1 steps to your west, sand 2 steps to your south-west. You face grass at your front. Your inventory is 1 wood pickaxe.
"""
FEEDBACKS = ""