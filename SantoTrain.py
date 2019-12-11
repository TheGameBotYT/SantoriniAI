from SantoGame import SantoriniEnv
from SantoAI import QLearningAgent, ReplayBuffer
from SantoAI import play_and_train, play_and_train_with_replay
import time
import pickle

rb = ReplayBuffer(size=10000)

# SantoQ.p -> training against a uniform policy opponent

env = SantoriniEnv()
agent = QLearningAgent(lr=0.1, gamma=0.99, epsilon=1,
                       get_legal_actions=env.get_viable_actions, filepath=None)
env.opponent_agent = agent

for i in range(0, 500000):
    if i % 10**4 == 0:
        print(i)
    """
    if (i % 10**5== 0) & (i!=0):
        file_name = 'Q' + str(int(i/10**5)) + '00krb16X.p'
        start = time.time()
        agent.output_to_pickle(file_name)
        end = time.time()
        print('Time to output:', end-start)
    """
    if i == 20*10**4:
        agent.epsilon = 0.5
    if i == 30*10**4:
        agent.epsilon = 0.3
    if i == 40*10**4:
        agent.epsilon = 0.2
    if i == 45*10**4:
        agent.epsilon = 0.05
    play_and_train_with_replay(env, agent, replay=rb, t_max=10**4, replay_batch_size=16)
    # play_and_train(env, agent, t_max=10**4)

start = time.time()
agent.output_to_pickle('QDerpo500krb16.p')
end = time.time()


