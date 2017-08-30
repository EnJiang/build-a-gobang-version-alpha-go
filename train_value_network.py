import math
import time
import random
import copy
import os
import os.path

import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
import pymysql
from concurrent.futures import ProcessPoolExecutor, wait
import simplejson


class Board(object):
    def __init__(self):
        self.height = 15
        self.width = 15
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

    def state_feature(self, last_action = None):
        width = self.width
        height = self.height

        feature_n = 17
        feature = []
        for position, one in enumerate(self.state):  # 主循环，生成一个点的feature，除法都是data scale
            feature_i = []
            feature_i.append(one / 2) # 棋点状态

            up_count, up_block, down_count, down_block = self.vertical_line_check(position)
            if (up_count == 0 and down_count == 0):
                vertical_count = 0
            else:
                vertical_count = up_count + down_count - 1
            feature_i.append(vertical_count / 5)
            feature_i.append(up_block)
            feature_i.append(down_block)

            left_count, left_block, right_count, right_block = self.horizontal_line_check(position)
            if (left_count == 0 and right_count == 0):
                horizontal_count = 0
            else:
                horizontal_count = left_count + right_count - 1
            feature_i.append(horizontal_count / 5)
            feature_i.append(left_block)
            feature_i.append(right_block)

            up_right_count, up_right_block, down_left_count, down_left_block = self.up_right_diagonal_check(position)
            if (up_right_count == 0 and down_left_count == 0):
                up_right_diagonal_count = 0
            else:
                up_right_diagonal_count = up_right_count + down_left_count - 1
            feature_i.append(up_right_diagonal_count / 5)
            feature_i.append(up_right_block)
            feature_i.append(down_left_block)

            up_left_count, up_left_block, down_right_count, down_right_block = self.up_left_diagonal_check(position)
            if (up_left_count == 0 and down_right_count == 0):
                up_left_diagonal_count = 0
            else:
                up_left_diagonal_count = up_left_count + down_right_count - 1
            feature_i.append(up_left_diagonal_count / 5)
            feature_i.append(up_left_block)
            feature_i.append(down_right_block)

            if last_action == None:
                h_now = 0
                w_now = 0
                h_last = 0
                w_last = 0
            else:
                h_now = position // width
                w_now = position % width
                h_last = last_action // width
                w_last = last_action % width
            last_action_distance = math.sqrt((h_now - h_last) ** 2 + (w_now - w_last) ** 2)
            scaled_distance = last_action_distance / (14 * math.sqrt(2))
            feature_i.append(round(scaled_distance, 3))

            count_1, count_2, count_blank = self.neighbour_count(position)
            feature_i.append(round(count_1 / 20, 3))
            feature_i.append(round(count_2 / 20, 3))
            feature_i.append(round(count_blank / 20, 3))

            feature.append(feature_i)

        return np.array(feature).reshape((15, 15, feature_n)).tolist()

    def vertical_line_check(self, position):
        width = self.width
        height = self.height
        size = width * height

        origin_pos = position

        position = origin_pos
        up_count = 0 # 向上检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            up_count = 0
            up_block = -1
        else:
            while 1:
                up_count += 1
                if position - width < 0: # 在顶行
                    up_block = 1
                    break
                if self.state[position - width] == state: # 如果上一行是同色，继续
                    position = position - width
                else: # 否则检查有没有被挡，结束
                    if self.state[position - width] != -1: # 是异色
                        up_block = 1
                    else:
                        up_block = 0
                    break

        position = origin_pos
        down_count = 0 # 向下检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            down_count = 0
            down_block = -1
        else:
            while 1:
                down_count += 1
                if position + width >= size: # 在底行
                    down_block = 1
                    break
                if self.state[position + width] == state: # 如果下一行是同色，继续
                    position = position + width
                    continue
                else: # 否则检查有没有被挡，结束
                    if self.state[position + width] != -1: # 异色
                        down_block = 1
                    else:
                        down_block = 0
                    break
        return up_count, up_block, down_count, down_block

    def horizontal_line_check(self, position):
        width = self.width
        height = self.height
        size = width * height

        origin_pos = position

        position = origin_pos
        left_count = 0 # 向左检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            left_count = 0
            left_block = -1
        else:
            while 1:
                left_count += 1
                if position % 15 == 0: # 在最左行
                    left_block = 1
                    break
                if self.state[position - 1] == state: # 如果左边是同色，继续
                    position = position - 1
                    continue
                else: # 否则检查有没有被挡，结束
                    if self.state[position - 1] != -1: # 异色
                        left_block = 1
                    else:
                        left_block = 0
                    break

        position = origin_pos
        right_count = 0 # 向右检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            right_count = 0
            right_block = -1
        else:
            while 1:
                right_count += 1
                if (position + 1) % 15 == 0: # 在最右行
                    right_block = 1
                    break
                if self.state[position + 1] == state: # 如果下一行是同色，继续
                    position = position + 1
                    continue
                else: # 否则检查有没有被挡，结束
                    if self.state[position + 1] != -1: # 异色
                        right_block = 1
                    else:
                        right_block = 0
                    break
        return left_count, left_block, right_count, right_block

    def up_right_diagonal_check(self, position):
        width = self.width
        height = self.height
        size = width * height

        origin_pos = position

        position = origin_pos
        up_right_count = 0 # 向右上检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            up_right_count = 0
            up_right_block = -1
        else:
            while 1:
                up_right_count += 1
                if position - width < 0 or (position + 1) % 15 == 0:  # 在顶行或者最右行
                    up_right_block = 1
                    break
                if self.state[position - width + 1] == state: # 如果右上方是同色，继续
                    position = position - width + 1
                else: # 否则检查有没有被挡，结束
                    if self.state[position - width + 1] != -1: # 是异色
                        up_right_block = 1
                    else:
                        up_right_block = 0
                    break

        position = origin_pos
        down_left_count = 0 # 向左下检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            down_left_count = 0
            down_left_block = -1
        else:
            while 1:
                down_left_count += 1
                if position + width >= size or position % 15 == 0: # 在底行或者最左行
                    down_left_block = 1
                    break
                if self.state[position + width - 1] == state: # 如果左下方是同色，继续
                    position = position + width - 1
                    continue
                else: # 否则检查有没有被挡，结束
                    if self.state[position + width - 1] != -1: # 异色
                        down_left_block = 1
                    else:
                        down_left_block = 0
                    break
        return up_right_count, up_right_block, down_left_count, down_left_block

    def up_left_diagonal_check(self, position):
        width = self.width
        height = self.height
        size = width * height

        origin_pos = position

        position = origin_pos
        up_left_count = 0 # 向左上检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            up_left_count = 0
            up_left_block = -1
        else:
            while 1:
                up_left_count += 1
                if position - width < 0 or position  % 15 == 0:  # 在顶行或者最左行
                    up_left_block = 1
                    break
                if self.state[position - width - 1] == state: # 如果左上方是同色，继续
                    position = position - width - 1
                else: # 否则检查有没有被挡，结束
                    if self.state[position - width - 1] != -1: # 是异色
                        up_left_block = 1
                    else:
                        up_left_block = 0
                    break

        position = origin_pos
        down_right_count = 0 # 向右下检查相连
        state = self.state[position] # 棋点颜色（或空）
        if state == -1: # 空盘，不检查
            down_right_count = 0
            down_right_block = -1
        else:
            while 1:
                down_right_count += 1
                if position + width >= size or (position + 1) % 15 == 0: # 在底行或者最右行
                    down_right_block = 1
                    break
                if self.state[position + width + 1] == state: # 如果右下方是同色，继续
                    position = position + width + 1
                    continue
                else: # 否则检查有没有被挡，结束
                    if self.state[position + width + 1] != -1: # 异色
                        down_right_block = 1
                    else:
                        down_right_block = 0
                    break
        return up_left_count, up_left_block, down_right_count, down_right_block

    def neighbour_count(self, position):
        width = self.width
        height = self.height

        origin_state = self.state[position]

        h = position // width
        w = position % width

        count_dict = {1: 0, 2: 0, -1: 0}

        for check_pos, state in enumerate(self.state):
            check_h = check_pos // width
            check_w = check_pos % width
            if math.fabs(check_h - h) <= 3 and math.fabs(check_w - w) <= 3:
                count_dict[state] += 1
        return count_dict[1], count_dict[2], count_dict[-1]

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

        children = [self.record.find(node) for node in possible_children]

        result = self.model.predict(np.array([np.array(node.state).reshape((15, 15, 1)) for node in possible_children]))

        value, node = max(
            [node.x_i + v_s[0], node] for node, v_s in zip(children, result)
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
        self.model = self.__model__()

    def __model__(self):
        model = Sequential()

        model.add(Convolution2D(512, (2, 2), border_mode='same',
                              input_shape=(15, 15, 17)))   # 15 - 3
        model.add(Activation('relu'))
        model.add(Convolution2D(256, (2, 2), border_mode='same'))   # 15 - 4
        model.add(Activation('relu'))
        model.add(Convolution2D(256, (2, 2), border_mode='valid'))  # 14 - 0
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (2, 2), border_mode='same'))  # 14 - 1
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (2, 2), border_mode='same'))  # 14 -2
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), border_mode='valid'))  # 12 - 0
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 6 - 0

        model.add(Convolution2D(64, (3, 3), border_mode='valid')) # 4 - 0
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 2 - 0

        # # 64 * 2 * 2 = 256
        #
        model.add(Dropout(0.25))
        #
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        #
        adam = Adam(lr=10e-7)
        model.compile(loss='binary_crossentropy',
                       optimizer=adam, metrics=['mae'])

        model.save("random_train_changed_weight.model")
        model.summary()

        # model = load_model("random_train_changed_weight.model")
        # model.summary()
        # plot_model(model, to_file="test.jpg", show_shapes=True, show_layer_names=False)

        return model

    def check_player(self, state):
        player1_count = state.count(1)
        player2_count = state.count(2)
        # 1 first, so if equal, return 1
        return 2 if player1_count > player2_count else 1


    def train_from_csv(self, csv_file):
        tesorboard = TensorBoard(write_graph=False)

        f = open(csv_file, "r")
        f.readline()
        line = f.readline()
        model = self.model
        x = []
        y = []
        while line:
            if len(x) == 10000:
                model.fit(x, y, nb_epoch=1, shuffle=True, verbose=1, batch_size=32,
                          callbacks=[tesorboard])
                model.save("random_train_changed_weight.model")
                x = []
                y = []

            item_raw = line.split(';')
            state = simplejson.loads(item_raw[0][1:-1])
            good_for_1 = float(item_raw[1])

            x.append(state)
            y.append(good_for_1)

            line = f.readline()

    def random_play(self):
        feature_n = 17
        feature_list = []

        self.board.clear()
        copy_board = copy.deepcopy(self.board)
        feature_list.append(copy_board.state_feature())

        # rand process
        current_player = 1
        over, winner = copy_board.check_game_process()
        player_id = current_player
        action = None
        while not over:
            # self.graphic(copy_board)
            # self.feature_graphic(copy_board.state_feature(last_action), 1)
            last_action = action
            action = random.choice(copy_board.legal_list)
            copy_board.update(player_id, action)
            feature_list.append(copy_board.state_feature(last_action))
            over, winner = copy_board.check_game_process()
            player_id = self.check_player(copy_board.state)
        # rand process
        # self.graphic(copy_board)
        # self.feature_graphic(copy_board.state_feature(last_action), 1)

        if winner == 1:
            y = 0.75
        else:
            y = 0.25

        x = feature_list
        num = len(x)
        x = np.array(x).reshape(num, 15, 15, feature_n)

        return x, y

    def random_test(self):
        model = self.model
        self.board.clear()

        # rand process
        current_player = 1
        over, winner = self.board.check_game_process()
        player_id = current_player
        action = None
        round = 1
        while not over:
            if player_id == 1:
                last_action = action

                current_root = Node(self.board.state)
                board = Board()
                board.set_with_state(current_root.state)
                legal_action_list = board.legal_list

                children_features = []
                possible_children = []
                for action in legal_action_list:
                    board.set_with_state(current_root.state)
                    player_id = self.check_player(board.state)
                    board.update(player_id, action)
                    feature = board.state_feature(last_action)
                    child = Node(board.state)
                    children_features.append(feature)
                    possible_children.append(child)
                result = model.predict(children_features)
                print("player %d now, " % player_id, end="")
                print("should pick a %s one." % ("highest" if player_id == 1 else "lowest"), end="")
                print("predicted avgerage wining posibility for player 1: %f" % (
                sum([one[0] for one in result]) / len(result)))
                self.graphic_with_predict(self.board, result, round)
                self.board.set_with_state(possible_children[np.argmax(result)].state)

            if player_id == 2:
                action = self.ai.rand_act()
                self.board.update(2, action)
            # copy_board.update(player_id, action)
            over, winner = self.board.check_game_process()
            player_id = self.check_player(self.board.state)
            # rand process

            round = round + 1

        self.graphic_with_predict(self.board, result, round)
        return winner

    def random_test_pre(self):
        win = 0
        game_count = 0
        while game_count != 1000:
            winner = self.random_test()
            if winner == 1:
                win += 1
            game_count += 1
            print(game_count, win / game_count)

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

    def graphic(self, board=None):
        if board == None:
            board = self.board
        state_list = board.state_graphic

        print("%-6s" % "", end="")
        for i in range(self.board.width):
            print("%-6s" % (str(i)), end="")
        print('\n')

        for i, line in enumerate(state_list):
            print("%-6s" % (str(i)), end="")
            for one in line:
                if one == 1:
                    ch = 'O'
                elif one == 2:
                    ch = 'X'
                else:
                    ch = '_'
                print("%-6s" % ch, end="")
            print('\n')

    def feature_graphic(self, feature, n):
        # print(feature)
        state_list = feature

        print("%-6s" % "", end="")
        for i in range(self.board.width):
            print("%-6s" % (str(i)), end="")
        print('\n')

        for i, line in enumerate(state_list):
            print("%-6s" % (str(i)), end="")
            for one in line:
                ch = one[n]
                print("%-6s" % str(ch)[:5], end="")
            print('\n')

    def game_over(self):
        over, winner = self.board.check_game_process()
        self.winner = winner
        return over

    def test_predict(self):
        model = self.model

        self.board.clear()
        copy_board = copy.deepcopy(self.board)

        # rand process
        current_player = 1
        over, winner = copy_board.check_game_process()
        player_id = current_player
        last_action = None
        while not over:
            current_root = Node(copy_board.state)
            board = Board()
            board.set_with_state(current_root.state)
            legal_action_list = board.legal_list

            children_features = []
            for action in legal_action_list:
                board.set_with_state(current_root.state)
                player_id = self.check_player(board.state)
                board.update(player_id, action)
                feature = board.state_feature(last_action)
                children_features.append(feature)

            result = model.predict(children_features)

            print("player %d now, " % player_id, end="")
            print("should pick a %s one." % ("highest" if player_id == 1 else "lowest"), end="")
            print("predicted avgerage wining posibility for player 1:", end="")
            self.graphic_with_predict(copy_board, result)

            # action = random.choice(copy_board.legal_list)
            action = self.human.act()
            last_action = action
            copy_board.update(player_id, action)
            over, winner = copy_board.check_game_process()
            player_id = self.check_player(copy_board.state)
        # rand process

    def graphic_with_predict(self, board, predict, round):
        # print(predict)
        copy_board = copy.deepcopy(board)
        legal_action_list = copy_board.legal_list
        for i, action in enumerate(legal_action_list):
            copy_board.state[action] = predict[i][0]

        state_list = copy_board.state_graphic

        x = []
        y = []
        c = []

        for i, line in enumerate(state_list):
            for j, one in enumerate(line):
                if one == 1:
                    plt.scatter(i, j, c='black', s=100, marker='o', edgecolors='black')
                elif one == 2:
                    plt.scatter(i, j, c='white', s=100, marker='X', edgecolors='black')
                else:
                    x.append(i)
                    y.append(j)
                    c.append(one)
        # print(c)
        cm = plt.cm.get_cmap('RdYlBu')
        plt.scatter(x, y, c=c, cmap=cm)
        plt.colorbar()
        plt.savefig("pic/%s.png" % str(round))
        plt.clf()

    def generate_train_data(self):
        x, y = self.random_play()
        db = pymysql.connect("localhost", "root", "123000000z", "n_in_a_row", charset='utf8')
        cur = db.cursor()
        stmt = "INSERT INTO play VALUES(%s, %s)"
        x = x.tolist()
        for one in x[:-1]:
            data = [str(one), str(y)]
            cur.execute(stmt, data)
        data = [str(x[:-1]), 1 if y else 0]
        cur.execute(stmt, data)
        db.commit()
        db.close()

if __name__ == "__main__":


    def inner_loop():
        game = Game()
        game.generate_train_data()
        # rootdir = "D:\\5_in_row\\"
        # while True:
        #     for parent, dirnames, filenames in os.walk(rootdir):
        #         for filename in filenames:
        #             # print(rootdir + filename)
        #             game.train_from_csv(rootdir + filename)

    inner_loop()

    # game = Game()
    # game.test_predict()
    # game.random_test()
    # game.random_test_pre()

    # while True:
    #     missions = []
    #     executor = ProcessPoolExecutor(32)
    #     for _ in range(1000):
    #         missions.append(executor.submit(inner_loop))
    #     wait(missions, return_when='ALL_COMPLETED')