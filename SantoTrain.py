from SantoGame import SantoriniEnv
from SantoAI import QLearningAgent, ReplayBuffer
from SantoAI import play_and_train, play_and_train_with_replay
import time

rb = ReplayBuffer(size=1000)

# SantoQ.p -> training against a uniform policy opponent

env = SantoriniEnv()
agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=1, get_legal_actions=env.get_viable_actions)
env.opponent_agent = agent

for i in range(1):
    if i % 1000 == 0:
        print(i)
    if i == 2*10**4:
        agent.epsilon = 0.5
    if i == 4*10**4:
        agent.epsilon = 0.2
    if i == 6*10**4:
        agent.epsilon = 0.1
    play_and_train_with_replay(env, agent, replay=rb, t_max=10**4, replay_batch_size=2048)
    # play_and_train(env, agent, t_max=10**4)

start = time.time()
agent.output_to_pickle('Qrb16.p')
end = time.time()
print(len(rb._storage))
print(rb._storage)


