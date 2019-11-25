from collections import defaultdict
import numpy as np
import pandas as pd
import time

"""
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
"""

import random
from collections import deque


class ReplayBuffer(object):
    def __init__(self, size):
        """
        Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.

        Note: for this assignment you can pick any data structure you want.
              If you want to keep it simple, you can store a list of tuples of (s, a, r, s') in self._storage
              However you may find out there are faster and/or more memory-efficient ways to do so.
        """
        self._storage = deque([])
        self._maxsize = size

        # OPTIONAL: YOUR CODE

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        '''
        Make sure, _storage will not exceed _maxsize.
        Make sure, FIFO rule is being followed: the oldest examples has to be removed earlier
        '''
        data = (obs_t, action, reward, obs_tp1, done)

        # add data to storage
        self._storage.append(data)
        if self.__len__() > self._maxsize:
            self._storage.popleft()

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # <batch_size> Number of random choices from self.__len__
        idxes = np.random.choice(np.arange(self.__len__()), batch_size)

        # collect <s,a,r,s',done> for each index
        states = []
        actions = []
        rewards = []
        next_states = []
        is_done = []
        for index in idxes:
            states.append(self._storage[index][0])
            actions.append(self._storage[index][1])
            rewards.append(self._storage[index][2])
            next_states.append(self._storage[index][3])
            is_done.append(self._storage[index][4])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(is_done)


class QLearningAgent(object):

    def __init__(self, lr, gamma, epsilon, get_legal_actions, filepath=None):
        # TODO: Implement this to continue from a saved Q-file of iterations
        if filepath is None:
            self._N = defaultdict(lambda: defaultdict(lambda: 0))
            self._Q = defaultdict(lambda: defaultdict(lambda: 0))
            self.get_legal_actions = get_legal_actions
            self.lr = lr
            self.gamma = gamma
            self.epsilon = epsilon
        else:
            self.get_legal_actions = get_legal_actions
            self._Q = pd.read_pickle(filepath)

    def update(self, state, action, reward, next_state):
        q_value = (1-self.lr) * self.get_qvalue(state, action) + \
            self.lr * (reward + self.gamma*self.get_value(next_state))

        self.set_qvalue(state, action, q_value)

    def get_qvalue(self, state, action):
        # TODO: This can't be efficient
        update_state = tuple([state[k] for k in sorted(state)])
        print(update_state)
        print(sorted(state))
        return self._Q[update_state][action]

    def set_qvalue(self, state, action, value):
        # TODO: This can't be efficient
        update_state = tuple([state[k] for k in sorted(state)])
        self._Q[update_state][action] = value

    def get_value(self, state):
        possible_actions = self.get_legal_actions()
        if len(possible_actions) == 0:
            return 0

        q_values = [self.get_qvalue(state, action) for action in possible_actions]
        value = np.max(q_values)
        return value

    def get_best_action(self, state):
        possible_actions = self.get_legal_actions()
        if len(possible_actions) == 0:
            return None

        try:
            q_values = [self.get_qvalue(state, action) for action in possible_actions]
            print(q_values)
            if all(v == 0 for v in q_values):
                best_action = np.random.choice(possible_actions)  # Else it will always pick first
            else:
                ind = np.argmax(q_values)
                best_action = possible_actions[ind]
        except KeyError:
            print('Never seen before, trying random')
            best_action = np.random.choice(possible_actions)
        return best_action

    def take_choice(self, state):
        possible_actions = self.get_legal_actions()
        if len(possible_actions) == 0:
            return None

        if np.random.uniform() < self.epsilon:
            chosen_action = np.random.choice(possible_actions)
        else:
            chosen_action = self.get_best_action(state)

        return chosen_action

    def output_to_pickle(self, filename):
        Q_table = pd.DataFrame.from_dict(self._Q)
        Q_table.to_pickle(filename)


def play_and_train(env, agent, t_max=10**4):
    total_reward = 0
    s = env.reset()
    for t in range(t_max):
        a = agent.take_choice(s)
        next_s, r, done = env.step(a)
        agent.update(s, a, r, next_s)
        s = next_s
        total_reward += r
        if done:
            break
    return total_reward


def play_and_train_with_replay(env, agent, replay=None, t_max=10**4, replay_batch_size=32):
    total_reward = 0
    s = env.reset()
    for t in range(t_max):
        a = agent.take_choice(s)
        next_s, r, done = env.step(a)
        agent.update(s, a, r, next_s)

        if replay is not None:
            replay.add(s, a, r, next_s, done)

            s_batch, a_batch, r_batch, next_s_batch, done_batch = replay.sample(replay_batch_size)
            for point in range(replay_batch_size):
                agent.update(s_batch[point], a_batch[point], r_batch[point], next_s_batch[point])

        s = next_s
        total_reward += r
        if done:
            break

    return total_reward

# TODO: Fix the table saving thing otherwise this will never work effectively, even when deep
# TODO: Change state space to LocationCurrentPlayer, LocationOpponentPlayer, Build States, helps with the table and now
# TODO: this trains player 2 to help player 1 win.
