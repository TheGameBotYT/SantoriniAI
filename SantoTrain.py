from SantoGame import SantoriniEnv
from SantoAI import QLearningAgent
from SantoAI import play_and_train

env = SantoriniEnv()
agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=0.1, get_legal_actions=env.get_viable_actions)

for i in range(500000):
    if i % 1000 == 0:
        print(i)
    play_and_train(env, agent, t_max=10**4)

agent.output_to_pickle('SantoQ.p')


