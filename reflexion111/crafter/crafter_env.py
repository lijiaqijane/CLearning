import json
from action import logger

class WrapEnv:
    def __init__(self, env):
        self._env = env
        self.taskname = "collect_diamond"
        self.subgoal = dict()
        self.achievements = {'collect_coal': 0, 'collect_diamond': 0, 'collect_drink': 0, 'collect_iron': 0,
                             'collect_sapling': 0, 'collect_stone': 0, 'collect_wood': 0, 'defeat_skeleton': 0,
                             'defeat_zombie': 0, 'eat_cow': 0, 'eat_plant': 0, 'make_iron_pickaxe': 0,
                             'make_iron_sword': 0, 'make_stone_pickaxe': 0, 'make_stone_sword': 0,
                             'make_wood_pickaxe': 0, 'make_wood_sword': 0, 'place_furnace': 0, 'place_plant': 0,
                             'place_stone': 0, 'place_table': 0, 'wake_up': 0}
        self.done = 0
        self.previous_action = []
        self.previous_observation = ''
        self.achieve_subgoal = []

    def set_task(self, task):
        self._env.reset()
        self.taskname = task
        # output
        with open('/scratch/nlp/lijiaqi/RAM_PPO/reflexion/crafter/subgoals.json', 'r') as file:
            subgoals = json.load(file)
        try:
            taskname = type(eval(self.taskname))
            self.subgoal = subgoals
        except:
            self.subgoal = subgoals[self.taskname]

    def cutoutObs(self, desc):
        index_start = desc.find("Your status:")
        index_end = desc.find("energy:") + 13
        index_prestart = desc.find("You see:")
        head = desc[index_prestart:index_start]
        tail = desc[index_end:]
        return head + tail

    def convert_achievements_to_list(self, achievements):
        achi_list = [[task, achievements[task]] for task in achievements]
        return achi_list

    def steps(self, actions_list):
        self.previous_action.extend(actions_list)
        indexs = [int(action.split('.')[0]) for action in actions_list]
        decs = list()

        for i in indexs:
            print(i)
            _, reward, done, info = self._env.step(i)
            obs = info['obs'].replace('\n',' ')   #self.cutoutObs(info['obs']).replace('\n',' ')
            decs.append(obs)
            self.done = done
            self.achievements = info['achievements']
  
        return decs[0], reward

    def subgoals_progress(self, all=False):
        if all:
            achievements = self.convert_achievements_to_list(self.achievements)
            return achievements
        else:
            prog = list()
            for task, (task_name, num) in enumerate(self.subgoal):
                prog.append([task_name, (self.achievements[task_name], num)])
            print(prog)


    def get_executable_actions(self):
        return {1: 'Move West', 2: 'Move East', 3: 'Move North', 4: 'Move South',
                                      5: 'Do',
                                      6: 'Sleep', 7: 'Place Stone',
                                      8: 'Place Table', 9: 'Place Furnace', 10: 'Place Plant', 11: 'Make Wood Pickaxe',
                                      12: 'Make Stone Pickaxe', 13: 'Make Iron Pickaxe', 14: 'Make Wood Sword',
                                      15: 'Make Stone Sword',
                                      16: 'Make Iron Sword'}