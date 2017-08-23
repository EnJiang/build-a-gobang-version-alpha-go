import math
import time
import random
import copy
import numpy as np


class Board(object):
    def __init__(self):
        self.height = 8
        self.width = 8
        self.n_in_row = 5
        self.state = [-1 for _ in range(self.height * self.width)]
        self.legal_list = [i for i in range(self.height * self.width)]

    def clear(self):
        self.state = [-1 for _ in range(self.height * self.width)]
        self.legal_list = [i for i in range(self.height * self.width)]

    def set_with_state(self, state):
        self.state = copy.copy(state)
        self.legal_list = [i for i in range(self.height * self.width)]
        for i, one in enumerate(state):
            if one != -1:
                self.legal_list.remove(i)

    def update(self, player_id, action):
        self.state[action] = player_id
        self.legal_list.remove(action)

    @property
    def state_graphic(self):
        state_array = np.array(self.state)
        state_array = state_array.reshape((self.height, self.width))
        return state_array.tolist()

    def check_game_process(self):
        '''
        :return: over: bool value, is game over?
        winner: the winner id, -1 for game is not over or draw
        '''
        occupied_pos_list = list(set(range(self.width * self.height)) - set(self.legal_list))

        if(len(occupied_pos_list) < self.n_in_row + 2):
            return False, -1

        width = self.width
        height = self.height
        state = self.state
        n = self.n_in_row
        for m in occupied_pos_list:
            h = m // width
            w = m % width
            player = state[m]

            if (w in range(width - n + 1) and
                    len(set(state[i] for i in range(m, m + n))) == 1):  # 横向连成一线
                return True, player

            if (h in range(height - n + 1) and
                    len(set(state[i] for i in range(m, m + n * width, width))) == 1):  # 竖向连成一线
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(state[i] for i in range(m, m + n * (width + 1), width + 1))) == 1):  # 右斜向上连成一线
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(state[i] for i in range(m, m + n * (width - 1), width - 1))) == 1):  # 左斜向下连成一线
                return True, player

        if len(self.legal_list):
            return False, -1
        else:
            return True, -1


class Player(object):
    def __init__(self, board):
        self.board = board

    def set_id(self, id):
        self.id = id

    @property
    def legal_action_list(self):
        return self.board.legal_list


class Node(object):
    def __init__(self, state):
        self.state = copy.copy(state)
        self.win_i = 0
        self.n_i = 0

    def __lt__(self, other):
        return True

    @property
    def x_i(self):
        return self.win_i / self.n_i

    def ucb(self, n):
        return (self.win_i / self.n_i) + math.sqrt(2 * math.log(n) / self.n_i)


class Record(object):
    def __init__(self):
        self.n = 0
        self.nodes = {}

    def find(self, node):
        try:
            return self.nodes[str(node.state)]
        except:
            return None

    def add(self, node):
        self.nodes[str(node.state)] = node

    def clear(self):
        self.nodes = {}


class AI(Player):
    def __init__(self, board):
        Player.__init__(self, board)
        self.record = Record()

    def act(self, think_time=30):
        action = self.mcts(think_time)
        return action

    def mcts(self, think_time):
        # find the postion of current state(and) play in the tree,
        # result is logically guaranteed to be found
        self.record.clear()
        current_root = Node(self.board.state)
        self.record.add(current_root)
        current_root = self.record.find(current_root)

        board = Board()
        board.set_with_state(current_root.state)
        legal_action_list = board.legal_list

        possible_children = []
        for action in legal_action_list:
            board.set_with_state(current_root.state)
            player_id = self.check_player(board.state)
            board.update(player_id, action)
            child = Node(board.state)
            possible_children.append(child)

        for child in possible_children:
            self.record.add(child)
            child_inside = self.record.find(child)
            tmp_board = Board()
            tmp_board.set_with_state(child_inside.state)
            favourable = self.simulate(board)
            if favourable:
                child_inside.win_i += 1
            child_inside.n_i += 1
            self.record.n += 1

        children = [self.record.find(node) for node in possible_children]

        self.simulation_count = 0
        start_time = time.time()
        while time.time() - start_time < think_time:
            one, _ = self.select(current_root)
            self.simulation_count += 1
            tmp_board = Board()
            tmp_board.set_with_state(one.state)
            favourable = self.simulate(tmp_board)
            if favourable:
                one.win_i += 1
            one.n_i += 1
            self.record.n += 1

        value, node = max(
            [node.x_i, node] for node in children
        )


        diff = (np.array(current_root.state) - np.array(node.state)).tolist()
        print(diff)
        action = diff.index(min(diff))

        print("simulation_count:", self.simulation_count)
        return action

    def select(self, current_node):
        board = Board()
        board.set_with_state(current_node.state)
        legal_action_list = board.legal_list

        possible_children = []
        for action in legal_action_list:
            board.set_with_state(current_node.state)
            player_id = self.check_player(board.state)
            board.update(player_id, action)
            child = Node(board.state)
            possible_children.append(child)

        children = [self.record.find(node) for node in possible_children]
        if(all(children)):
            value, node = max([
                [node.ucb(self.record.n), node] for node in children
            ])
            return node, False
        else:
            return None, True

    def simulate(self, board):
        # self.graphic(board)
        copy_board = copy.deepcopy(board)
        current_player = self.check_player(copy_board.state)

        # rand process
        over, winner = copy_board.check_game_process()
        player_id = current_player
        while not over:
            action = random.choice(copy_board.legal_list)
            copy_board.update(player_id, action)
            over, winner = copy_board.check_game_process()
            player_id = self.check_player(copy_board.state)
        # rand process

        # print(board.state)

        return winner != current_player

    def check_player(self, state):
        player1_count = state.count(1)
        player2_count = state.count(2)
        # 1 first, so if equal, return 1
        return 2 if player1_count > player2_count else 1

    def rand_act(self):
        legal_action_list = self.legal_action_list

        return random.choice(legal_action_list)


    def __str__(self):
        return "AI"


class Human(Player):
    def __init__(self, board):
        Player.__init__(self, board)

    def act(self, think_time = None):
        print("your move:")
        cmd = input()
        action = self.parse(cmd)
        while not (action in self.board.legal_list):
            print("invalid input")
            cmd = input()
            action = self.parse(cmd)
        else:
            return action



    def parse(self, cmd):
        width = self.board.width
        try:
            y, x = map(int, cmd.split(','))
            return y * width + x
        except:
            return -1


    def __str__(self):
        return "Human"


class Game(object):
    def __init__(self):
        self.player1 = None
        self.player2 = None
        self.board = Board()
        self.ai = AI(self.board)
        self.human = Human(self.board)

    def play_with_human(self):
        start_time = time.time()
        self.player1 = self.ai
        self.player2 = self.human

        rand_order = random.choice([[1, 2], [2, 1]])
        self.player1.set_id(rand_order[0])
        self.player2.set_id(rand_order[1])
        play_order = [self.player1, self.player2] if self.player1.id == 1 else \
            [self.player2, self.player1]

        self.board.clear()
        while not self.game_over():
            self.graphic()
            current_player = play_order[0]
            action = current_player.act(think_time=90)
            self.board.update(current_player.id, action)
            play_order.reverse()

        self.graphic()
        print("%s player wins, id %d" % (str(current_player), current_player.id) )

    def graphic(self):
        state_list = self.board.state_graphic

        print("%-3s" % "", end="")
        for i in range(self.board.width):
            print("%-3s" % (str(i)), end="")
        print('\n')

        for i, line in enumerate(state_list):
            print("%-3s" % (str(i)), end="")
            for one in line:
                if one == 1:
                    ch = 'O'
                elif one == 2:
                    ch = 'X'
                else:
                    ch = '_'
                print("%-3s" % ch, end="")
            print('\n')

    def game_over(self):
        over, winner = self.board.check_game_process()
        self.winner = winner
        return over


if __name__ == "__main__":
    game = Game()
    game.play_with_human() # play with human