import numpy as np


class SantoriniEnv(object):

    def __init__(self, opponent_agent=None, mode='learn'):
        # TODO: Perhaps use reset() as an initial state setting
        self.mode = mode
        self.num_builders = 1
        self.size = 3
        self.player_positions = {1: None, 2: None}
        if opponent_agent is None:
            self.opponent_agent = self.dummy_opponent_agent
        else:
            self.opponent_agent = opponent_agent
        self.action_dict = {0: (1, 1), 1: (0, 1), 2: (-1, 1),
                            3: (1, 0), 4: (-1, 0),
                            5: (1, -1), 6: (0, -1), 7: (-1, -1)}
        self.inv_action_dict = {v: k for k, v in self.action_dict.items()}
        self.coords, self.inv_coords = self.load_coord_dict()
        self.reset()

    def dummy_opponent_agent(self, state):
        viable_actions = self.get_viable_actions()
        choice_int = np.random.choice(len(viable_actions))
        action = viable_actions[choice_int]
        return action

    def reset(self):
        self.state = self.init_state()
        self.init_player_positions()
        self.phase = 'Move'
        self.current_player = np.random.choice([1, 2])

    def init_player_positions(self):
        # TODO: Think of something to either make this a start choice or make sure there is no overlap
        # TODO: Should be choice 0 -> Place workers
        start_list = []
        for _ in range(0, self.num_builders):
            for player in range(1, 3):
                ind = np.random.randint(self.size**2)
                while ind in start_list:
                    ind = np.random.randint(self.size**2)
                start_list.append(ind)
                state_name = 'Move' + str(ind)
                self.state[state_name] = player
                self.player_positions[player] = ind

    def init_state(self):
        state = {}
        for x in range(self.size**2):
            move_state = 'Move' + str(x)
            build_state = 'Build' + str(x)
            state[move_state] = 0
            state[build_state] = 0
        return state

    def load_coord_dict(self):
        coord = {0: (0, 0), 1: (0, 1), 2: (0, 2),
                 3: (1, 0), 4: (1, 1), 5: (1, 2),
                 6: (2, 0), 7: (2, 1), 8: (2, 2)}
        inv_coord = {v: k for k, v in coord.items()}
        return coord, inv_coord

    def step(self, action):
        print('Action: ', action)
        viability, info = self.check_action_viability(action)
        print('-----------------')
        if not viability:
            print('NO VIABILITY', info)
            return self.state, 0, False
        else:
            new_s = self.evolve_state_given_action(action)
            done, reward = self.end_condition()
            if done:
                print('Reward: ', reward)
                print('End state: ', self.state)
                return new_s, reward, done
            while self.current_player == 2:
                a = self.opponent_agent(new_s)
                new_s = self.evolve_state_given_action(a)
                done, reward = self.end_condition()
                if done:
                    print('Reward: ', reward)
                    print('End state: ', self.state)
                    break
            return new_s, reward, done  # Possible Others?

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
                buildings += str(self.state[build_name])
                builders += '  '
                move_name = str('Move') + str(x + y*3)
                builders += str(self.state[move_name])
            buildings += '\n'
            builders += '\n'
        print(buildings)
        print(builders)

    def end_condition(self):  # Every so often the end of game condition is checked
        """
        Game ends when player places
        :return:
        """
        build_states = [(k, v) for k, v in self.state.items() if 'Build' in k]
        for (k, v) in build_states:
            if v == 3:
                state_name = 'Move' + k.lstrip('Build')
                if self.state[state_name] == 1:
                    return True, 1
                elif self.state[state_name] == 2:
                    return True, -1
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
        if self.state[state_name] == 4:
            info = "Dome at target position"
            return False, info
        state_name = 'Move' + str(target_pos)
        if self.state[state_name] > 0:
            info = "Builder there"
            return False, info
        if self.phase == 'Move':
            target_state = 'Build' + str(target_pos)
            self_state = 'Build' + str(current_pos)
            if self.state[target_state] > (self.state[self_state] + 1):
                info = "Too high to climb"
                return False, info
        info = "Viable"
        return True, info

    def action_given_target_position(self, target_pos):
        current_pos = self.get_position_coord()
        x, y = self.coords[current_pos]
        x2, y2 = self.coords[target_pos]
        delta = (x2-x, y2-y)
        action = self.inv_action_dict[delta]
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
        """
        :param action: tuple of (player, 'Builder'/'Building', X move, Y move)
        """
        target_pos = self.inv_coords[self.target_position_given_action(action)]
        current_pos = self.get_position_coord()
        if self.phase == 'Move':  # Move
            current_state = 'Move' + str(current_pos)
            target_state = 'Move' + str(target_pos)
            self.state[current_state] = 0  # Remove from current position
            self.state[target_state] = self.current_player  # Stand on target position
            self.player_positions[self.current_player] = target_pos
            self.phase = 'Build'  # Change to other phase
        elif self.phase == 'Build':  # Build
            target_state = 'Build' + str(target_pos)
            self.state[target_state] += 1
            self.phase = 'Move'  # Change to other phase
            self.determine_next_player()
        return self.state

    def determine_next_player(self):
        if self.current_player == 1:
            self.current_player = 2
        elif self.current_player == 2:
            self.current_player = 1

