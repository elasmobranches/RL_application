import numpy as np
import random
from sudoku_env import SudokuEnv # 에이전트는 환경의 내부 구조를 직접 알 필요가 줄어듦

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000):
        # self.env = env # 환경 인스턴스를 직접 가질 필요가 없을 수 있음
        self.lr = learning_rate
        self.gamma = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        if epsilon_decay_steps > 0:
            self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
        else:
            self.epsilon_decay_rate = 0
        
        self.q_table = {} # {agent_state_tuple: {chosen_number_action: q_value}}
                        # agent_state_tuple = ((r,c), possible_nums_tuple)

    def get_q_value(self, agent_state, chosen_number):
        """주어진 상태(현재 빈칸 좌표, 가능한 숫자들)에서 특정 숫자를 선택했을 때의 Q-값을 반환합니다.
        이 함수는 호출하는 쪽에서 agent_state가 None이 아님을 보장한다고 가정합니다.
        """
        return self.q_table.get(agent_state, {}).get(chosen_number, 0.0)

    def choose_action(self, agent_state):
        """엡실론-그리디 정책에 따라 현재 상태에서 행동(놓을 숫자)을 선택합니다.
        agent_state는 ((row, col), possible_nums_tuple) 형태입니다.
        possible_nums_tuple이 비어있으면 None을 반환합니다 (더 이상 놓을 수 있는 수가 없음).
        """
        if agent_state is None or not agent_state[1]: # 상태가 없거나, 현재 빈칸에 가능한 숫자가 없으면
            return None 

        _, possible_nums = agent_state

        if not possible_nums: # 이중 체크, 위에서 이미 agent_state[1]로 체크했지만 명시적으로
            return None

        if random.uniform(0, 1) < self.epsilon:
            # 탐험: 가능한 숫자 중 무작위 선택
            action_chosen_number = random.choice(possible_nums)
        else:
            # 활용: 가장 높은 Q-값을 가진 숫자 선택
            q_values = {num: self.get_q_value(agent_state, num) for num in possible_nums}
            max_q = -float('inf')
            # Q-값이 같은 경우를 대비해, 최대 Q값을 가진 액션들을 모음
            best_actions = []
            for num, q_val in q_values.items():
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [num]
                elif q_val == max_q:
                    best_actions.append(num)
            
            action_chosen_number = random.choice(best_actions) # 최대 Q-값을 가진 액션들 중 무작위 선택
        
        return action_chosen_number

    def update_q_value(self, agent_state, chosen_number, reward, next_agent_state):
        """Q-테이블을 업데이트합니다.
        agent_state, next_agent_state는 ((row, col), possible_nums_tuple) 형태입니다.
        chosen_number는 agent_state에서 선택된 숫자(액션)입니다.
        """
        if agent_state is None:
            return # 이전 상태가 없으면 업데이트 불가

        old_q_value = self.get_q_value(agent_state, chosen_number)
        
        next_max_q = 0.0
        if next_agent_state is not None and next_agent_state[1]: # 다음 상태가 있고, 가능한 숫자가 있다면
            next_possible_nums = next_agent_state[1]
            if next_possible_nums: # 다음 상태에서 가능한 액션(숫자)들이 있다면
                q_values_next_state = [self.get_q_value(next_agent_state, num) for num in next_possible_nums]
                if q_values_next_state: # Q 값들이 존재하면 (빈 리스트가 아니면)
                    next_max_q = max(q_values_next_state)
        
        # Q-learning 업데이트 규칙
        new_q_value = old_q_value + self.lr * (reward + self.gamma * next_max_q - old_q_value)
        
        # Q-테이블 업데이트
        if agent_state not in self.q_table:
            self.q_table[agent_state] = {}
        self.q_table[agent_state][chosen_number] = new_q_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end and self.epsilon_decay_steps > 0:
            self.epsilon -= self.epsilon_decay_rate
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end

if __name__ == '__main__':
    # QLearningAgent 테스트 코드 (기본적인 구조만 확인)
    env_test = SudokuEnv(grid_size=4)
    agent_test = QLearningAgent(epsilon_decay_steps=1000)

    example_puzzle_q = [
        [1, 0, 3, 0],
        [0, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 0, 4]
    ]
    env_test.reset(puzzle=example_puzzle_q) # 보드 초기화
    current_agent_state_test = env_test._get_current_agent_state() # 에이전트가 이해하는 상태 형태 가져오기

    print(f"Initial epsilon: {agent_test.epsilon}")
    action_test = agent_test.choose_action(current_agent_state_test) # 올바른 상태 전달
    print(f"Chosen action: {action_test}")

    if action_test:
        # env.step은 다음 에이전트 상태를 올바른 형태로 반환
        next_agent_state_test, reward_test, done_test, _ = env_test.step(action_test)
        agent_test.update_q_value(current_agent_state_test, action_test, reward_test, next_agent_state_test) # 올바른 상태 전달
        print(f"Q-value for (state, action): {agent_test.get_q_value(current_agent_state_test, action_test)}")

    agent_test.decay_epsilon()
    print(f"Epsilon after decay: {agent_test.epsilon}")
    print(f"Q-table (sample): {list(agent_test.q_table.items())[:2]}") # Q-테이블 일부 출력 