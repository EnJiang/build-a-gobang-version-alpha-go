import math
import time
import random

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

    def update(self, player, action):
        self.state[action] = player.id
        self.legal_list.remove(action)

    @property
    def state_graphic(self):
        state_array = np.array(self.state)
        state_array = state_array.reshape((self.height, self.width))
        return state_array.tolist()



    def check_game_process(self):
        occupied_pos_list = list(set([i for i in range(self.height * self.width)]) - set(self.legal_list))
        for one in occupied_pos_list:
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

        return False, -1




class Player(object):
    def __init__(self, board):
        self.board = board

    def set_id(self, id):
        self.id = id


    def legal_action_list(self, state):
        legal_list = []
        for i, one in enumerate(state):
            if one == -1:
                legal_list.append(i)
        return legal_list


class AI(Player):
    def __init__(self, board):
        Player.__init__(self, board)


    def act(self, state):
        legal_action_list = self.legal_action_list(state)

        return random.choice(legal_action_list)


    def __str__(self):
        return "AI"


class Human(Player):
    def __init__(self, board):
        Player.__init__(self, board)

    def act(self, state):
        cmd = input()
        action = self.parse(cmd)
        return action


    def parse(self, cmd):
        width = self.board.width
        try:
            y, x = map(int, cmd.split(','))
            return y * width + x
        except:
            print("invalid input")


    def __str__(self):
        return "Human"


class Game(object):
    def __init__(self):
        self.player1 = None
        self.player2 = None
        self.board = Board()
        self.ai = AI(self.board)
        self.human = Human(self.board)

    def self_train(self, train_time=1000):
        start_time = time.time()
        self.player1 = self.ai
        self.player2 = self.ai
        while time.time() - start_time < train_time:
            self.play(graphic=False)
            print(time.time() - start_time)

    def play_with_human(self):
        start_time = time.time()
        self.player1 = self.ai
        self.player2 = self.human
        self.play(graphic=True)

    def play(self, graphic=True):
        rand_order = random.choice([[1, 2], [2, 1]])
        self.player1.set_id(rand_order[0])
        self.player2.set_id(rand_order[1])
        play_order = [self.player1, self.player2] if self.player1.id == 1 else \
            [self.player2, self.player1]

        self.board.clear()
        while not self.game_over():
            if graphic:
                self.graphic()
            current_player = play_order[0]
            action = current_player.act(self.board.state)
            self.board.update(current_player, action)
            play_order.reverse()
        if graphic:
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
    # game.self_train()  # self-play train for 1000sec
    game.play_with_human()  # play with human