from SantoGame import SantoriniEnv

import numpy as np

env = SantoriniEnv()

states_list, action_list, reward_list, done_list = [], [], [], []

print(env.player_positions)
done = False
while not done:
    env.render()
    states_list.append(env.state)
    viable_actions = env.get_viable_actions()
    choice_int = np.random.choice(len(viable_actions))
    action = viable_actions[choice_int]
    new_s, reward, done = env.step(action)
    action_list.append(action)
    reward_list.append(reward)
    done_list.append(done)
env.render()
print(states_list)
print(action_list)
print(reward_list)
print(done_list)