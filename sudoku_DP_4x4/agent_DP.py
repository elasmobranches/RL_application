import numpy as np
import collections
import random
from sudoku_env_DP import SudokuEnv

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.policy = {}
        self.value_table = collections.defaultdict(float)

    def policy_iteration(self, initial_state):
        reachable_states = self._get_reachable_states(initial_state)
        for state_tuple in reachable_states:
            possible_actions = self.env.get_possible_actions(state_tuple)
            if possible_actions:
                self.policy[state_tuple] = random.choice(possible_actions)
            else:
                self.policy[state_tuple] = None
        policy_stable = False
        while not policy_stable:
            self._policy_evaluation(reachable_states) # self.policy에 저장된 정책을 사용하여 각 상태의 가치를 계산
            policy_stable = self._policy_improvement(reachable_states) # 각 상태에 대해 가치가 최대가 되는 액션을 선택하여 정책 갱신

    def _get_reachable_states(self, initial_state):
        queue = collections.deque([initial_state])
        visited = {initial_state}
        while queue:
            current_state_tuple = queue.popleft()
            board = self.env.get_board_from_state(current_state_tuple)
            if len(self.env.get_empty_cells_from_board(board)) == 0:
                continue
            possible_actions = self.env.get_possible_actions(current_state_tuple)
            for action in possible_actions:
                temp_board = board.copy()
                r, c = action[0]
                num = action[1]
                temp_board[r, c] = num
                next_state_tuple = tuple(temp_board.flatten())
                if next_state_tuple not in visited:
                    visited.add(next_state_tuple)
                    queue.append(next_state_tuple)
        return list(visited)

    def _policy_evaluation(self, states, theta=1e-8):
        while True:
            delta = 0
            for state_tuple in states:
                old_value = self.value_table[state_tuple]
                action = self.policy.get(state_tuple)
                new_value = self._calculate_value(state_tuple, action) # 현재 상태에서 선택된 액션의 가치를 계산
                self.value_table[state_tuple] = new_value # 현재 상태의 가치를 업데이트
                delta = max(delta, abs(old_value - new_value)) # 가치 변동 최대값 계산
            if delta < theta: # 가치 변동 최대값이 임계값보다 작으면 종료
                break

    def _policy_improvement(self, states):
        policy_stable = True
        for state_tuple in states:
            old_action = self.policy.get(state_tuple)
            possible_actions = self.env.get_possible_actions(state_tuple)
            if not possible_actions:
                continue
            action_values = {action: self._calculate_value(state_tuple, action) for action in possible_actions}
            best_action = max(action_values, key=action_values.get)
            self.policy[state_tuple] = best_action
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    def _calculate_value(self, state_tuple, action):
        if action is None:
            board = self.env.get_board_from_state(state_tuple)
            if len(self.env.get_empty_cells_from_board(board)) == 0:
                return 1.0 if SudokuEnv.is_fully_solved(board) else -1.0
            return -1.0
        r, c = action[0]
        num = action[1]
        board = self.env.get_board_from_state(state_tuple)
        temp_env = SudokuEnv(board=board)
        _, reward, _ = temp_env.step(action)
        next_state_tuple = tuple(temp_env.board.flatten())
        return reward + self.gamma * self.value_table[next_state_tuple]

    def get_policy(self, state_tuple):
        return self.policy.get(state_tuple)

class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-8):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_table = collections.defaultdict(float)

    def value_iteration(self, initial_state):
        reachable_states = self._get_reachable_states(initial_state)
        for state in reachable_states:
            board = self.env.get_board_from_state(state)
            if len(self.env.get_empty_cells_from_board(board)) == 0:
                if SudokuEnv.is_fully_solved(board):
                    self.value_table[state] = 1.0
                else:
                    self.value_table[state] = -1.0
            else:
                self.value_table[state] = 0.0
        while True:
            delta = 0
            for state in reachable_states:
                board = self.env.get_board_from_state(state)
                possible_actions = self.env.get_possible_actions(state)
                if not possible_actions:
                    continue
                action_values = []
                for action in possible_actions:
                    temp_env = SudokuEnv(board=board)
                    _, reward, _ = temp_env.step(action)
                    next_state = tuple(temp_env.board.flatten())
                    value = reward + self.gamma * self.value_table[next_state]
                    action_values.append(value)
                new_value = max(action_values)
                delta = max(delta, abs(self.value_table[state] - new_value))
                self.value_table[state] = new_value
            if delta < self.theta:
                break

    def _get_reachable_states(self, initial_state):
        queue = collections.deque([initial_state])
        visited = {initial_state}
        while queue:
            current_state_tuple = queue.popleft()
            board = self.env.get_board_from_state(current_state_tuple)
            if len(self.env.get_empty_cells_from_board(board)) == 0:
                continue
            possible_actions = self.env.get_possible_actions(current_state_tuple)
            for action in possible_actions:
                temp_board = board.copy()
                r, c = action[0]
                num = action[1]
                temp_board[r, c] = num
                next_state_tuple = tuple(temp_board.flatten())
                if next_state_tuple not in visited:
                    visited.add(next_state_tuple)
                    queue.append(next_state_tuple)
        return list(visited)

    def get_policy(self, state_tuple):
        board = self.env.get_board_from_state(state_tuple)
        if len(self.env.get_empty_cells_from_board(board)) == 0:
            return None
        possible_actions = self.env.get_possible_actions(state_tuple)
        if not possible_actions:
            return None
        action_values = {}
        for action in possible_actions:
            temp_env = SudokuEnv(board=board)
            _, reward, _ = temp_env.step(action)
            next_state = tuple(temp_env.board.flatten())
            value = reward + self.gamma * self.value_table[next_state]
            action_values[action] = value
        best_action = max(action_values.items(), key=lambda x: x[1])[0]
        return best_action 