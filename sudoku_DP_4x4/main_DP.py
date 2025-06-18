from sudoku_env_DP import SudokuEnv
from agent_DP import ValueIterationAgent, PolicyIterationAgent
from utils_DP import print_board, TEST_PUZZLES_4X4, EXAMPLE_PUZZLES_4X4
import numpy as np
import time
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import os

# 파일 출력을 위한 클래스 - 터미널과 파일에 동시에 출력하기 위한 클래스
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

# 가치 함수를 히트맵으로 시각화하고 저장하는 함수
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

# 현재 상태에서 가능한 모든 액션에 대한 가치를 그리드 형태로 출력하는 함수
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

def plot_value_comparison(agent, initial_state, env, title_prefix=""):
    """
    각 grid(빈 칸)에 대해 가능한 action 중 value가 가장 큰 action(숫자, value)을 찾아
    value_grid에는 value, action_grid에는 숫자를 저장하고,
    시각화 텍스트에는 '숫자(value)' 형태로 표시합니다.
    디버깅을 위해 각 (i, j) 위치별로 가능한 action의 num, value, best_value, best_num을 print합니다.
    """
    save_dir = "value_comparison"
    os.makedirs(save_dir, exist_ok=True)
    
    max_value_board = env.get_board_from_state(agent.max_value_state)
    final_board = env.get_board_from_state(initial_state)
    
    max_value_grid = np.zeros((4, 4))
    max_action_grid = np.zeros((4, 4), dtype=int)
    final_value_grid = np.zeros((4, 4))
    final_action_grid = np.zeros((4, 4), dtype=int)
    
    # 최대 value 상태: 각 칸별 최적 action/value
    possible_actions = env.get_possible_actions(agent.max_value_state)
    for i in range(4):
        for j in range(4):
            if max_value_board[i, j] == 0:
                best_value = float('-inf')
                best_num = 0
                for action in possible_actions:
                    (row, col), num = action
                    if row == i and col == j:
                        if isinstance(agent, ValueIterationAgent):
                            temp_env = SudokuEnv(board=max_value_board)
                            _, reward, _ = temp_env.step(action)
                            next_state = tuple(temp_env.board.flatten())
                            value = reward + agent.gamma * agent.value_table[next_state]
                        else:
                            value = agent._calculate_value(agent.max_value_state, action)
                        print(f"[DEBUG] MaxState ({i},{j}) num={num}, value={value:.3f}, best_value={best_value:.3f}, best_num={best_num}")
                        if value > best_value:
                            best_value = value
                            best_num = num
                print(f"[DEBUG] MaxState 최종 선택: 위치({i},{j}) -> 숫자 {best_num}, value {best_value:.3f}")
                max_value_grid[i, j] = best_value
                max_action_grid[i, j] = best_num
            else:
                max_value_grid[i, j] = 0.0
                max_action_grid[i, j] = max_value_board[i, j]
    
    # 최종 상태: 각 칸별 최적 action/value
    possible_actions = env.get_possible_actions(initial_state)
    for i in range(4):
        for j in range(4):
            if final_board[i, j] == 0:
                best_value = float('-inf')
                best_num = 0
                for action in possible_actions:
                    (row, col), num = action
                    if row == i and col == j:
                        if isinstance(agent, ValueIterationAgent):
                            temp_env = SudokuEnv(board=final_board)
                            _, reward, _ = temp_env.step(action)
                            next_state = tuple(temp_env.board.flatten())
                            value = reward + agent.gamma * agent.value_table[next_state]
                        else:
                            value = agent._calculate_value(initial_state, action)
                        print(f"[DEBUG] FinalState ({i},{j}) num={num}, value={value:.3f}, best_value={best_value:.3f}, best_num={best_num}")
                        if value > best_value:
                            best_value = value
                            best_num = num
                print(f"[DEBUG] FinalState 최종 선택: 위치({i},{j}) -> 숫자 {best_num}, value {best_value:.3f}")
                final_value_grid[i, j] = best_value
                final_action_grid[i, j] = best_num
            else:
                final_value_grid[i, j] = 0.0
                final_action_grid[i, j] = final_board[i, j]
    
    eng_title_prefix = title_prefix.replace("학습 단계", "Learning Step").replace("최대 Value 상태", "Max Value State").replace("최종 상태", "Final State").replace("Value Iteration", "Value Iteration").replace("Policy Iteration", "Policy Iteration")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    im1 = ax1.imshow(max_value_grid, cmap='RdYlGn', vmin=-20, vmax=20)
    ax1.set_title(f"{eng_title_prefix} Max Value State\n(Value: {agent.max_value:.2f})")
    plt.colorbar(im1, ax=ax1)
    
    im2 = ax2.imshow(final_value_grid, cmap='RdYlGn', vmin=-20, vmax=20)
    ax2.set_title(f"{eng_title_prefix} Final State")
    plt.colorbar(im2, ax=ax2)
    
    for i in range(4):
        for j in range(4):
            # Max value state: 숫자(value) 형태로 표시
            if max_value_board[i, j] == 0:
                text_color = "white" if max_value_grid[i, j] > 0 else "black"
                ax1.text(j, i, f"{max_action_grid[i, j]}({max_value_grid[i, j]:.2f})", ha="center", va="center", color=text_color)
            else:
                ax1.text(j, i, str(max_value_board[i, j]), ha="center", va="center", color="gray")
            # Final state: 숫자(value) 형태로 표시
            if final_board[i, j] == 0:
                text_color = "white" if final_value_grid[i, j] > 0 else "black"
                ax2.text(j, i, f"{final_action_grid[i, j]}({final_value_grid[i, j]:.2f})", ha="center", va="center", color=text_color)
            else:
                ax2.text(j, i, str(final_board[i, j]), ha="center", va="center", color="gray")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{eng_title_prefix}_value_comparison.png"))
    plt.close()

def print_value_comparison_ascii(agent, initial_state, env, title_prefix=""):
    """
    최대 value 상태와 최종 상태의 value를 ASCII 아트로 터미널에 출력합니다.
    """
    # 최대 value 상태와 최종 상태의 보드 가져오기
    max_value_board = env.get_board_from_state(agent.max_value_state)
    final_board = env.get_board_from_state(initial_state)
    
    # 두 상태의 value 그리드 생성
    max_value_grid = np.zeros((4, 4))
    final_value_grid = np.zeros((4, 4))
    
    # 최대 value 상태의 value 계산
    possible_actions = env.get_possible_actions(agent.max_value_state)
    for action in possible_actions:
        (row, col), num = action
        if isinstance(agent, ValueIterationAgent):
            temp_env = SudokuEnv(board=max_value_board)
            _, reward, _ = temp_env.step(action)
            next_state = tuple(temp_env.board.flatten())
            value = reward + agent.gamma * agent.value_table[next_state]
        else:  # PolicyIterationAgent
            value = agent._calculate_value(agent.max_value_state, action)
        max_value_grid[row, col] = value
    
    # 최종 상태의 value 계산
    possible_actions = env.get_possible_actions(initial_state)
    for action in possible_actions:
        (row, col), num = action
        if isinstance(agent, ValueIterationAgent):
            temp_env = SudokuEnv(board=final_board)
            _, reward, _ = temp_env.step(action)
            next_state = tuple(temp_env.board.flatten())
            value = reward + agent.gamma * agent.value_table[next_state]
        else:  # PolicyIterationAgent
            value = agent._calculate_value(initial_state, action)
        final_value_grid[row, col] = value
    
    print(f"\n=== {title_prefix} Value 비교 ===")
    print(f"최대 Value: {agent.max_value:.2f}")
    
    # 최대 value 상태 출력
    print("\n[최대 Value 상태]")
    print("보드:")
    for i in range(4):
        if i > 0 and i % 2 == 0:
            print("-" * 13)
        for j in range(4):
            if j > 0 and j % 2 == 0:
                print("|", end=" ")
            print(f"{max_value_board[i,j] if max_value_board[i,j] != 0 else '.'}", end=" ")
        print()
    
    print("\nValue:")
    for i in range(4):
        if i > 0 and i % 2 == 0:
            print("-" * 25)
        for j in range(4):
            if j > 0 and j % 2 == 0:
                print("|", end=" ")
            if max_value_board[i,j] == 0:
                print(f"{max_value_grid[i,j]:.2f}", end=" ")
            else:
                print("  -  ", end=" ")
        print()
    
    # 최종 상태 출력
    print("\n[최종 상태]")
    print("보드:")
    for i in range(4):
        if i > 0 and i % 2 == 0:
            print("-" * 13)
        for j in range(4):
            if j > 0 and j % 2 == 0:
                print("|", end=" ")
            print(f"{final_board[i,j] if final_board[i,j] != 0 else '.'}", end=" ")
        print()
    
    print("\nValue:")
    for i in range(4):
        if i > 0 and i % 2 == 0:
            print("-" * 25)
        for j in range(4):
            if j > 0 and j % 2 == 0:
                print("|", end=" ")
            if final_board[i,j] == 0:
                print(f"{final_value_grid[i,j]:.2f}", end=" ")
            else:
                print("  -  ", end=" ")
        print()
    print()

def print_all_action_values(agent, state_tuple, env, title=""):
    board = env.get_board_from_state(state_tuple)
    possible_actions = env.get_possible_actions(state_tuple)
    print(f"\n=== {title} 모든 action별 value ===")
    for i in range(4):
        for j in range(4):
            if board[i, j] == 0:
                print(f"위치 ({i},{j}):")
                for action in possible_actions:
                    (row, col), num = action
                    if row == i and col == j:
                        if isinstance(agent, ValueIterationAgent):
                            temp_env = SudokuEnv(board=board)
                            _, reward, _ = temp_env.step(action)
                            next_state = tuple(temp_env.board.flatten())
                            value = reward + agent.gamma * agent.value_table[next_state]
                        else:
                            value = agent._calculate_value(state_tuple, action)
                        print(f"  - 숫자 {num}: value = {value:.3f}")
    print()

# 주어진 퍼즐을 해결하는 함수
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

# 에이전트를 점진적으로 학습시키는 함수
def train_agent_progressive(agent_class, agent_name):
    print(f"\n=== {agent_name} 점진적 학습 시작 ===")
    env = SudokuEnv()
    agent = agent_class(env, gamma=0.99)
    total_learn_time = 0

    for i, p_data in enumerate(EXAMPLE_PUZZLES_4X4):
        # 퍼즐마다 max_value, max_value_state, value_history 초기화
        agent.max_value = float('-inf')
        agent.max_value_state = None
        agent.value_history = {}

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
        print(f"- 최대 value: {agent.max_value:.2f}")
        print_value_grid(agent, initial_state, env, f"{agent_name} 학습 단계 {i+1} 완료")
        
        # DEBUG 정보 출력
        # print(f"[DEBUG] Step {i+1} max_value_state: {agent.max_value_state}")
        # print(f"[DEBUG] Step {i+1} max_value: {agent.max_value}")
        # print(f"[DEBUG] Step {i+1} initial_state: {initial_state}")
        
        # 모든 action별 value 표 출력
        print_all_action_values(agent, initial_state, env, f"{agent_name} Step {i+1}")
        
        # value 비교 시각화 (이미지 파일로 저장)
        plot_value_comparison(agent, initial_state, env, f"{agent_name} 학습 단계 {i+1}")
        # value 비교 ASCII 출력
        print_value_comparison_ascii(agent, initial_state, env, f"{agent_name} 학습 단계 {i+1}")

    print(f"\n{agent_name} 전체 학습 완료:")
    print(f"- 총 학습 시간: {total_learn_time:.2f}초")
    return agent, total_learn_time

# 메인 실행 부분
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