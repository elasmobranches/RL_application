from sudoku_env_DP import SudokuEnv
from agent_DP import ValueIterationAgent, PolicyIterationAgent
from utils_DP import print_board, TEST_PUZZLES_4X4, EXAMPLE_PUZZLES_4X4
import numpy as np
import time
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import os

# 파일 출력을 위한 클래스
class FileOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_value_heatmap(value_grid, filename, title=""):
    # value_heatmap 폴더 생성
    save_dir = "value_heatmap"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    plt.figure(figsize=(4,4))
    cmap = plt.cm.RdYlGn
    plt.imshow(value_grid, cmap=cmap, vmin=-20, vmax=20)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            # 배경색에 따라 텍스트 색상 동적 변경
            text_color = "white" if value_grid[i, j] > 0 else "black"
            plt.text(j, i, f"{value_grid[i, j]:.2f}", ha="center", va="center", color=text_color)
    
    # 영어로 제목 변경
    if "Value Iteration" in title:
        title = title.replace("Value Iteration", "Value Iteration")
    elif "Policy Iteration" in title:
        title = title.replace("Policy Iteration", "Policy Iteration")
    if "학습 단계" in title:
        title = title.replace("학습 단계", "Learning Step")
    if "완료" in title:
        title = title.replace("완료", "")
    if "Test Puzzle" in title:
        title = title.replace("Test Puzzle", "Test Puzzle")
    if "Final" in title:
        title = title.replace("Final", "Final")
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def print_value_grid(agent, state_tuple, env, title=""):
    """
    현재 상태에서 가능한 모든 액션에 대한 가치를 그리드 형태로 출력합니다.
    """
    board = env.get_board_from_state(state_tuple)
    possible_actions = env.get_possible_actions(state_tuple)
    
    # 4x4 그리드 생성
    value_grid = np.zeros((4, 4))
    action_grid = np.zeros((4, 4), dtype=int)
    
    # 각 위치에 대한 가치 계산
    for action in possible_actions:
        (row, col), num = action
        if isinstance(agent, ValueIterationAgent):
            temp_env = SudokuEnv(board=board)
            _, reward, _ = temp_env.step(action)
            next_state = tuple(temp_env.board.flatten())
            value = reward + agent.gamma * agent.value_table[next_state]
        else:  # PolicyIterationAgent
            value = agent._calculate_value(state_tuple, action)
        
        value_grid[row, col] = value
        action_grid[row, col] = num
    
    print(f"\n=== {title} 가치 함수 그리드 ===")
    print("위치별 가치 (숫자는 가능한 액션):")
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:  # 빈 셀인 경우에만 가치 표시
                print(f"({i},{j}): {value_grid[i,j]:.3f} [{action_grid[i,j]}]", end="\t")
            else:
                print(f"({i},{j}): {board[i,j]}", end="\t")
        print()
    print()
    # 히트맵 이미지 저장
    save_value_heatmap(value_grid, f"{title.replace(' ', '_')}_value_heatmap.png", title)

def solve_puzzle(agent, initial_board, algo_name="", puzzle_idx=None):
    env = SudokuEnv(board=initial_board)
    current_board = env.reset()
    
    print("--- Initial Puzzle ---")
    print_board(current_board)
    
    max_steps = np.count_nonzero(current_board == 0) + 5
    step = 0
    solve_start_time = time.time()
    
    while len(env.get_empty_cells()) > 0 and step < max_steps:
        print(f"--- Step {step + 1} ---")
        current_state_tuple = env.get_state()
        
        action = agent.get_policy(current_state_tuple)
        
        if action is None:
            print("No action returned from policy. Puzzle might be unsolvable or in a dead-end.")
            break
            
        (row, col), num = action
        print(f"Action: Place {num} at ({row}, {col})")
        
        current_board, reward, done = env.step(action)
        
        print_board(current_board)
        
        if done:
            break
        step += 1

    solve_time = time.time() - solve_start_time
    print("--- Final Result ---")
    print_board(current_board)
    
    # 퍼즐 번호와 알고리즘명을 파일명에 포함 (영어로 title)
    final_state = tuple(current_board.flatten())
    if puzzle_idx is not None and algo_name:
        title = f"{algo_name} Test Puzzle {puzzle_idx+1} Final"
    else:
        title = "Final State"
    print_value_grid(agent, final_state, env, title)
    
    if SudokuEnv.is_fully_solved(current_board):
        print("Success! Puzzle solved.")
        return True, solve_time
    else:
        print("Failed to solve the puzzle.")
        return False, solve_time

def train_agent_progressive(agent_class, agent_name):
    print(f"\n=== {agent_name} 점진적 학습 시작 ===")
    env = SudokuEnv()
    agent = agent_class(env, gamma=0.99)
    total_learn_time = 0

    for i, p_data in enumerate(EXAMPLE_PUZZLES_4X4):
        print(f"\n--- {agent_name} 학습 단계 {i+1} ({p_data['difficulty']}) ---")
        print("학습 퍼즐:")
        print_board(p_data['puzzle'])
        initial_state = tuple(p_data['puzzle'].flatten())
        learn_start_time = time.time()
        if agent_name == "Value Iteration":
            agent.value_iteration(initial_state)
        else:
            agent.policy_iteration(initial_state)
        learn_time = time.time() - learn_start_time
        total_learn_time += learn_time
        print(f"학습 완료:")
        print(f"- 학습 시간: {learn_time:.2f}초")
        print(f"- 학습된 상태 수: {len(agent.value_table) if hasattr(agent, 'value_table') else len(agent.policy)}")
        print_value_grid(agent, initial_state, env, f"{agent_name} 학습 단계 {i+1} 완료")

    print(f"\n{agent_name} 전체 학습 완료:")
    print(f"- 총 학습 시간: {total_learn_time:.2f}초")
    return agent, total_learn_time

if __name__ == '__main__':
    # 출력 파일 설정
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'sudoku_DP_output_{timestamp}.txt'
    sys.stdout = FileOutput(output_file)
    
    print(f"\n실행 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # 각 알고리즘별로 점진적 학습
    vi_agent, vi_learn_time = train_agent_progressive(ValueIterationAgent, "Value Iteration")
    pi_agent, pi_learn_time = train_agent_progressive(PolicyIterationAgent, "Policy Iteration")
    
    total_vi_time = vi_learn_time
    total_pi_time = pi_learn_time
    total_vi_solve_time = 0
    total_pi_solve_time = 0
    
    # 성공한 퍼즐 수 카운트
    vi_success_count = 0
    pi_success_count = 0
    
    print("\n=== 테스트 퍼즐 해결 시작 ===")
    for i, data in enumerate(TEST_PUZZLES_4X4):
        puzzle = data['puzzle'].copy()
        print(f"\n===== 테스트 퍼즐 {i+1} ({data['difficulty']}) =====")
        
        # 학습된 정책으로 퍼즐 해결
        result_vi, vi_solve_time = solve_puzzle(vi_agent, puzzle, "ValueIteration", i)
        result_pi, pi_solve_time = solve_puzzle(pi_agent, puzzle, "PolicyIteration", i)
        
        # 성공한 퍼즐 카운트
        if result_vi:
            vi_success_count += 1
        if result_pi:
            pi_success_count += 1
        
        total_vi_time += vi_solve_time
        total_pi_time += pi_solve_time
        total_vi_solve_time += vi_solve_time
        total_pi_solve_time += pi_solve_time
        
        print(f"\n[테스트 퍼즐 {i+1} 요약]")
        print(f"Value Iteration: {'성공' if result_vi else '실패'} | Policy Iteration: {'성공' if result_pi else '실패'}")
        print(f"Value Iteration 해결 시간: {vi_solve_time:.2f}초")
        print(f"Policy Iteration 해결 시간: {pi_solve_time:.2f}초")
        print("=" * 50)
    
    total_puzzles = len(TEST_PUZZLES_4X4)
    print("\n=== 전체 성능 요약 ===")
    print(f"Value Iteration:")
    print(f"- 학습 시간: {vi_learn_time:.2f}초")
    print(f"- 평균 해결 시간: {total_vi_solve_time/total_puzzles:.2f}초")
    print(f"- 총 소요 시간: {total_vi_time:.2f}초")
    print(f"- 성공률: {vi_success_count}/{total_puzzles} ({vi_success_count/total_puzzles*100:.1f}%)")
    
    print(f"\nPolicy Iteration:")
    print(f"- 학습 시간: {pi_learn_time:.2f}초")
    print(f"- 평균 해결 시간: {total_pi_solve_time/total_puzzles:.2f}초")
    print(f"- 총 소요 시간: {total_pi_time:.2f}초")
    print(f"- 성공률: {pi_success_count}/{total_puzzles} ({pi_success_count/total_puzzles*100:.1f}%)")
    
    print(f"\n실행 종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n출력 파일이 저장되었습니다: {output_file}")
    
    # 원래 stdout으로 복구
    sys.stdout = sys.stdout.terminal 