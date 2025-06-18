import numpy as np
import collections
import random
from sudoku_env_DP import SudokuEnv

# 정책 반복(Policy Iteration) 알고리즘을 구현한 에이전트 클래스
class PolicyIterationAgent:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma  # 할인율(미래 보상에 대한 현재 가치)
        self.policy = {}    # 상태-행동 정책 테이블
        self.value_table = collections.defaultdict(float)  # 상태 가치 테이블
        self.value_history = {}  # value 변화 기록
        self.max_value_state = None  # 최대 value를 가진 상태
        self.max_value = float('-inf')  # 최대 value 값

    # 정책 반복 알고리즘의 메인 함수
    def policy_iteration(self, initial_state):
        # 모든 도달 가능한 상태 집합 생성
        reachable_states = self._get_reachable_states(initial_state)
        # 정책 초기화: 가능한 행동 중 무작위 선택
        for state_tuple in reachable_states:
            possible_actions = self.env.get_possible_actions(state_tuple)
            if possible_actions:
                self.policy[state_tuple] = random.choice(possible_actions)
            else:
                self.policy[state_tuple] = None
        policy_stable = False
        iteration = 0
        while not policy_stable:
            self._policy_evaluation(reachable_states) # 정책 평가 단계
            policy_stable = self._policy_improvement(reachable_states) # 정책 개선 단계
            
            # 현재 iteration의 value 기록
            current_values = {state: self.value_table[state] for state in reachable_states}
            self.value_history[iteration] = current_values
            
            # 최대 value 상태 업데이트
            for state, value in current_values.items():
                if value > self.max_value:
                    self.max_value = value
                    self.max_value_state = state
            
            iteration += 1

    # 주어진 초기 상태에서 도달 가능한 모든 상태를 찾는 함수 (BFS)
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

    # 현재 정책에 대한 상태 가치를 계산하는 함수
    def _policy_evaluation(self, states, theta=1e-8):
        while True:
            delta = 0
            for state_tuple in states:
                old_value = self.value_table[state_tuple]
                action = self.policy.get(state_tuple)
                new_value = self._calculate_value(state_tuple, action)
                self.value_table[state_tuple] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < theta:  # 가치 함수가 수렴하면 종료
                break

    # 현재 가치 함수를 기반으로 정책을 개선하는 함수
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

    # 주어진 상태와 행동에 대한 가치를 계산하는 함수
    def _calculate_value(self, state_tuple, action):
        if action is None:
            board = self.env.get_board_from_state(state_tuple)
            if len(self.env.get_empty_cells_from_board(board)) == 0:
                # 퍼즐이 완전히 채워졌을 때 정답이면 1, 아니면 -1
                return 1.0 if SudokuEnv.is_fully_solved(board) else -1.0
            return -1.0
        r, c = action[0]
        num = action[1]
        board = self.env.get_board_from_state(state_tuple)
        temp_env = SudokuEnv(board=board)
        _, reward, _ = temp_env.step(action)
        next_state_tuple = tuple(temp_env.board.flatten())
        return reward + self.gamma * self.value_table[next_state_tuple]

    # 주어진 상태에 대한 정책(행동)을 반환하는 함수
    def get_policy(self, state_tuple):
        return self.policy.get(state_tuple)

# 가치 반복(Value Iteration) 알고리즘을 구현한 에이전트 클래스
class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-8):
        self.env = env
        self.gamma = gamma  # 할인율
        self.theta = theta  # 수렴 판단을 위한 임계값
        self.value_table = collections.defaultdict(float)  # 상태 가치 테이블
        self.value_history = {}  # value 변화 기록
        self.max_value_state = None  # 최대 value를 가진 상태
        self.max_value = float('-inf')  # 최대 value 값

    # 가치 반복 알고리즘의 메인 함수
    def value_iteration(self, initial_state):
        reachable_states = self._get_reachable_states(initial_state)
        # 초기 가치 함수 설정
        for state in reachable_states:
            board = self.env.get_board_from_state(state)
            if len(self.env.get_empty_cells_from_board(board)) == 0:
                if SudokuEnv.is_fully_solved(board):
                    self.value_table[state] = 1.0
                else:
                    self.value_table[state] = -1.0
            else:
                self.value_table[state] = 0.0
        
        iteration = 0
        # 가치 반복
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
                
                # 최대 value 상태 업데이트
                if new_value > self.max_value:
                    self.max_value = new_value
                    self.max_value_state = state
            
            # 현재 iteration의 value 기록
            current_values = {state: self.value_table[state] for state in reachable_states}
            self.value_history[iteration] = current_values
            
            if delta < self.theta:  # 가치 함수가 수렴하면 종료
                break
            
            iteration += 1

    # 주어진 초기 상태에서 도달 가능한 모든 상태를 찾는 함수 (BFS)
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

    # 주어진 상태에 대한 최적 행동을 반환하는 함수
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