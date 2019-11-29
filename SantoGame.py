import numpy as np
import time

class DummyOpponent(object):

    def __init__(self, get_viable_actions_func):
        self.get_viable_actions = get_viable_actions_func

    def take_choice(self, state=None):
        viable_actions = self.get_viable_actions()
        choice_int = np.random.choice(len(viable_actions))
        action = viable_actions[choice_int]
        return action

class SantoriniEnv(object):

    def __init__(self, opponent_agent=None):
        self.num_builders = 1
        self.size = 3
        self.player_positions = {1: None, 2: None}
        self.phase = None
        self.current_player = 0
        self.state = []
        self.state_index_dict = {'Phase': 0}
        if opponent_agent is None:
            self.opponent_agent = DummyOpponent(self.get_viable_actions)
        else:
            self.opponent_agent = opponent_agent
        self.action_dict = {0: (1, 1), 1: (0, 1), 2: (-1, 1),
                            3: (1, 0), 4: (-1, 0),
                            5: (1, -1), 6: (0, -1), 7: (-1, -1)}
        self.phase_encode_dict = {'Move': 0, 'Build': 1}
        self.inv_action_dict = {v: k for k, v in self.action_dict.items()}
        self.coords, self.inv_coords = self.load_coord_dict()
        self.reset()

    def reset(self):
        self.phase = 'Move'
        self.state = self.init_state()
        self.current_player = np.random.choice([1, 2])
        self.init_player_positions()
        return self.state

    def init_player_positions(self):
        # TODO: Think of something to either make this a start choice
        # TODO: Should be choice 0 -> Place workers
        start_list = []
        for _ in range(0, self.num_builders):
            for player in range(1, 3):
                ind = np.random.randint(self.size**2)
                while ind in start_list:
                    ind = np.random.randint(self.size**2)
                start_list.append(ind)
                self.player_positions[player] = ind
        self.state[self.state_index_dict['SPos']] = self.player_positions[self.current_player]
        self.state[self.state_index_dict['OPos']] = self.player_positions[self.determine_other_player()]

    def init_state(self):
        state = [0] * (1 + 2 + self.size**2)
        self.state_index_dict['SPos'] = 1
        self.state_index_dict['OPos'] = 2
        for x in range(self.size**2):
            # move_state = 'Move' + str(x)
            build_state = 'Build' + str(x)
            # self.state_index_dict[move_state] = x + 1
            self.state_index_dict[build_state] = x + 3
            # state[self.state_index_dict[move_state]] = 0
            state[self.state_index_dict[build_state]] = 0
            state[self.state_index_dict['Phase']] = self.phase_encode_dict[self.phase]
        return state

    def load_coord_dict(self):
        coord = {0: (0, 0), 1: (0, 1), 2: (0, 2),
                 3: (1, 0), 4: (1, 1), 5: (1, 2),
                 6: (2, 0), 7: (2, 1), 8: (2, 2)}
        inv_coord = {v: k for k, v in coord.items()}
        return coord, inv_coord

    def step(self, action):
        # print('Player 1?', self.current_player)
        self.evolve_state_given_action(action)
        done, reward = self.end_condition()
        if done:
            return self.state, reward, done
        while self.current_player == 2:
            # print('Player 2?', self.current_player)
            a = self.opponent_agent.take_choice(self.state)
            self.evolve_state_given_action(a)
            done, reward = self.end_condition()
            if done:
                break
        return self.state, reward, done  # Possible Others?

    def step_play(self, action):
        if self.current_player == 1:
            viability, info = self.check_action_viability(action)
            if not viability:
                print('NO VIABILITY', info)
                return self.state, 0, False
            else:
                new_s = self.evolve_state_given_action(action)
                done, reward = self.end_condition()
        elif self.current_player == 2:
            # TODO: Rework this so dummy opponent still works
            a = self.opponent_agent.get_best_action(self.state)
            new_s = self.evolve_state_given_action(a)
            done, reward = self.end_condition()
        if done:
            print('Reward: ', reward)
            print('End state: ', self.state)
        return new_s, reward, done

    def render(self, mode='human', close=False):
        """
        Prints the game state in a 'nice' ASCII format
        :param mode:
        :param close:
        :return:
        """
        buildings = ''
        builders = ''
        for y in range(self.size):
            for x in range(self.size):
                buildings += '  '
                build_name = str('Build') + str(x+y*3)
                buildings += str(self.state[self.state_index_dict[build_name]])
                builders += '  '
                move_name = str('Move') + str(x + y*3)
                builders += str(self.state[self.state_index_dict[move_name]])
            buildings += '\n'
            builders += '\n'
        print(buildings)
        print(builders)

    def end_condition(self):  # Every so often the end of game condition is checked
        """
        Game ends when player places
        :return:
        """
        # TODO: More efficient to turn around,
        # Rather than, check each gridpoint if it is build3, then check player pos
        # Check player pos, then check if 3
        build_tups = [(k, v) for k, v in self.state_index_dict.items() if 'Build' in k]
        for k, v in build_tups:
            if self.state[v] == 3:
                state_name = k.lstrip('Build')
                if self.state[self.state_index_dict['SPos']] == int(state_name):
                    if self.current_player == 1:
                        return True, 1
                    elif self.current_player == 2:
                        return True, -1
        # TODO: Thoroughly check new implementation below:
        if len(self.get_viable_actions()) == 0:
            if self.current_player == 1:
                return True, -1
            elif self.current_player == 2:
                return True, 1
        return False, 0

    def check_action_viability(self, action):
        """
        Current implementation has the 8 directions in the action definition
        TODO: Requires something that forces a build after a move action
        :param action: tuple of (player, 'Builder'/'Building', X move, Y move)
        Steps:
        If move:
            1. Get current position
            2. Get target position using action
            3. Check if target position can be moved to, exceptions:
                i. Builder there
                ii. No building there with value > our value - 1 (can't climb that high)
                iii. Dome there
                iv. Not move off the board
        If build:
            1. Get current position
            2. Get target position using action
            3. Check if target can be built on, exceptions:
                i. Builder there
                ii. Dome there
                iii. Not build off the board
        :return: True or False depending on viable action
        """
        target_coord = self.target_position_given_action(action)
        current_pos = self.get_position_coord()
        try:
            target_pos = self.inv_coords[target_coord]
        except KeyError:
            info = "Out of bounds, coord does not exist"
            return False, info
        state_name = 'Build' + str(target_pos)
        if self.state[self.state_index_dict[state_name]] == 4:
            info = "Dome at target position"
            return False, info
        if target_pos == self.player_positions[self.determine_other_player()]:
            info = "Builder there"
            return False, info
        if self.phase == 'Move':
            target_state = 'Build' + str(target_pos)
            self_state = 'Build' + str(current_pos)
            if self.state[self.state_index_dict[target_state]] > (self.state[self.state_index_dict[self_state]] + 1):
                info = "Too high to climb"
                return False, info
        info = "Viable"
        return True, info

    def action_given_target_position(self, target_pos):
        current_pos = self.get_position_coord()
        x, y = self.coords[current_pos]
        x2, y2 = self.coords[target_pos]
        delta = (x2-x, y2-y)
        try:
            action = self.inv_action_dict[delta]
        except KeyError:
            print(delta, ' not a viable Position option')
            action = None
        return action

    def target_position_given_action(self, action):
        current_pos = self.get_position_coord()
        x, y = self.coords[current_pos]
        delta_tuple = self.action_dict[action]
        target_coord = (x + delta_tuple[0], y + delta_tuple[1])
        """
        if action == 0:
            target_coord = (x + 1, y + 1)
        elif action == 1:
            target_coord = (x + 0, y + 1)
        elif action == 2:
            target_coord = (x - 1, y + 1)
        elif action == 3:
            target_coord = (x + 1, y + 0)
        elif action == 4:
            target_coord = (x - 1, y + 0)
        elif action == 5:
            target_coord = (x + 1, y - 1)
        elif action == 6:
            target_coord = (x + 0, y - 1)
        elif action == 7:
            target_coord = (x - 1, y - 1)
        else:
            Warning('Hello')
        """
        return target_coord

    def get_viable_actions(self):
        """
        0 - Top Left
        1 - Above
        2 - Top Right
        3 - Left
        4 - Right
        5 - Bottom Left
        6 - Down
        7 - Bottom Right
        :return:
        """
        viable_actions = []
        for action in range(8):
            viability, info = self.check_action_viability(action)
            if viability:
                viable_actions.append(action)
        return viable_actions

    def get_position_coord(self):
        current_pos = self.player_positions[self.current_player]
        return current_pos

    def evolve_state_given_action(self, action):
        target_pos = self.inv_coords[self.target_position_given_action(action)]
        current_pos = self.get_position_coord()
        if self.phase == 'Move':  # Move
            # current_state = 'Move' + str(current_pos)
            # target_state = 'Move' + str(target_pos)
            # self.state[self.state_index_dict[current_state]] = 0  # Remove from current position
            # self.state[self.state_index_dict[target_state]] = self.current_player  # Stand on target position
            self.player_positions[self.current_player] = target_pos
            self.state[self.state_index_dict['SPos']] = target_pos
            self.phase = 'Build'  # Change to other phase
            self.state[self.state_index_dict['Phase']] = self.phase_encode_dict['Build']
        elif self.phase == 'Build':  # Build
            target_state = 'Build' + str(target_pos)
            self.state[self.state_index_dict[target_state]] += 1
            self.phase = 'Move'  # Change to other phase
            self.state[self.state_index_dict['Phase']] = self.phase_encode_dict['Move']
            self.current_player = self.determine_other_player()
            # Switch self, opponent when new player is determined
            spos_ind = self.state_index_dict['SPos']
            opos_ind = self.state_index_dict['OPos']
            self.state[spos_ind], self.state[opos_ind] = self.state[opos_ind], self.state[spos_ind]
        return self.state

    def determine_other_player(self):
        if self.current_player == 1:
            opp = 2
        elif self.current_player == 2:
            opp = 1
        return opp

