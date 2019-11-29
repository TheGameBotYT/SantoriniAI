from SantoGame import SantoriniEnv
from SantoAI import QLearningAgent, ReplayBuffer
from SantoAI import play_and_train, play_and_train_with_replay
import time

rb = ReplayBuffer(size=1000)

# SantoQ.p -> training against a uniform policy opponent

env = SantoriniEnv()
agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=1, get_legal_actions=env.get_viable_actions)
# env.opponent_agent = agent

for i in range(10000):
    if i % 1000 == 0:
        print(i)
    if i == 5*10**5:
        agent.epsilon = 0.5
    if i == 1*10**6:
        agent.epsilon = 0.2
    if i == 15*10**5:
        agent.epsilon = 0.1
    # play_and_train_with_replay(env, agent, replay=rb, t_max=10**4, replay_batch_size=32)
    play_and_train(env, agent, t_max=10**4)

start = time.time()
agent.output_to_pickle('Q20new.p')
end = time.time()
print('Time to output in seconds:', end-start)


