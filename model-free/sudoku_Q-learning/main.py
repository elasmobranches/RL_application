import numpy as np
import random
from sudoku_env import SudokuEnv
from q_learning_agent import QLearningAgent
from utils import EXAMPLE_PUZZLES_4X4, plot_rewards, get_random_puzzle_4x4

def train_agent(env, agent, episodes=10000, max_steps_per_episode=100, use_random_puzzles=True):
    episode_rewards = []
    solved_count = 0
    
    print(f"학습 시작: 총 {episodes} 에피소드, 에피소드 당 최대 스텝: {max_steps_per_episode}")
    if use_random_puzzles:
        print("매 에피소드마다 무작위 퍼즐을 사용합니다.")
    else:
        if EXAMPLE_PUZZLES_4X4:
            fixed_puzzle = np.array(EXAMPLE_PUZZLES_4X4[0])
            print(f"고정된 퍼즐로 학습합니다: {fixed_puzzle.tolist()}")
        else:
            print("고정 퍼즐이 없어 빈 보드로 학습합니다. (무작위 퍼즐 사용 권장)")
            fixed_puzzle = None
            use_random_puzzles = True

    for episode in range(episodes):
        if use_random_puzzles:
            puzzle = get_random_puzzle_4x4()
            if puzzle is None:
                _ = env.reset() # 반환값 사용 안함
            else:
                _ = env.reset(puzzle=puzzle.copy())
        else:
            if fixed_puzzle is not None:
                _ = env.reset(puzzle=fixed_puzzle.copy())
            else:
                _ = env.reset()
        
        current_agent_state = env._get_current_agent_state() # 새로운 상태 가져오기
        total_reward_for_episode = 0
        done = False
        
        for step_count in range(max_steps_per_episode):
            if current_agent_state is None: # 모든 셀이 채워짐
                # print(f"에피소드 {episode+1}, 스텝 {step_count}: 모든 셀이 채워져서 중단 (current_agent_state is None)")
                break 
            
            # current_agent_state[0]은 (r,c), current_agent_state[1]은 possible_nums
            if not current_agent_state[1]: # 현재 빈칸에 놓을 수 있는 숫자가 없는 경우 (막힌 상태)
                # print(f"에피소드 {episode+1}, 스텝 {step_count}: {current_agent_state[0]}에 가능한 숫자가 없어 중단")
                # 이 경우, 에이전트가 None을 반환하거나, 특정 페널티와 함께 에피소드 종료 가능
                # 여기서는 choose_action이 None을 반환하고 아래에서 처리되도록 함
                pass # 다음 agent.choose_action에서 None 반환 유도

            action_chosen_number = agent.choose_action(current_agent_state) # 상태 전달하여 숫자 선택
            
            if action_chosen_number is None: # 에이전트가 행동을 선택할 수 없는 경우 (possible_nums가 비었거나, epsilon 정책에 의해 탐험 안함 등)
                # print(f"에피소드 {episode+1}, 스텝 {step_count}: 에이전트가 행동을 선택하지 못함 (action_chosen_number is None)")
                # 이 경우 Q-테이블 업데이트 없이 에피소드 종료 또는 다음 스텝 진행 여부 결정 필요
                # 일반적으로 이런 상황은 막혔거나, 이미 푼 상태일 수 있음.
                # env.step 이전에 이 상황이 발생하면, 보상 부여가 애매해짐.
                # 만약 possible_nums가 비어서 action_chosen_number가 None이라면, 
                # 이전 step에서 이미 큰 페널티를 받았거나, is_solved()가 True여야 함.
                # 여기서는 루프를 중단. 이미 done = True 이거나, 환경 규칙상 막힌 경우임.
                break 
            
            # env.step은 선택된 숫자(action_chosen_number)를 인자로 받음
            next_agent_state, reward_from_env, done, info = env.step(action_chosen_number)
            
            if 'error' in info and "Invalid number" in info['error']:
                # step 함수 내부에서 유효하지 않은 숫자로 판명되어 페널티를 받고 done=True가 된 경우
                # Q-value 업데이트는 이뤄져야 함.
                pass


            agent.update_q_value(current_agent_state, action_chosen_number, reward_from_env, next_agent_state)
            
            current_agent_state = next_agent_state
            total_reward_for_episode += reward_from_env
            
            if done:
                # print(f"에피소드 {episode+1}, 스텝 {step_count}: done={done}, 보상={reward_from_env}, 정보={info}")
                break
        
        agent.decay_epsilon() 
        episode_rewards.append(total_reward_for_episode)
        
        if env.is_solved():
            solved_count += 1
            
        if (episode + 1) % (episodes // 20 if episodes >= 20 else 1) == 0:
            avg_reward_recent = np.mean(episode_rewards[-(episodes // 20 if episodes >= 20 else 1):])
            # recent_episodes_count = (episodes // 20 if episodes >= 20 else 1) # 이 변수는 아래 current_solve_rate 계산에 직접 사용되진 않음.
            
            # solved_count는 전체 에피소드에 대한 누적이므로, 최근 성공률을 보려면 별도 계산 필요 (이 주석은 일반적인 설명으로 유효)
            current_solve_rate = solved_count / (episode + 1)

            print(f"Episode {episode + 1}/{episodes} - 최근 평균 보상: {avg_reward_recent:.2f}, Epsilon: {agent.epsilon:.3f}, 누적 성공률: {current_solve_rate:.3f}")
                            
    print(f"학습 완료. 총 {episodes} 에피소드 중 {solved_count}번 해결 (성공률: {solved_count/episodes:.3f})")
    return episode_rewards, agent

def test_agent(env, agent, test_puzzles, num_tests=5):
    print("\n에이전트 테스트 시작...")
    solved_count = 0
    
    if not test_puzzles:
        print("테스트할 퍼즐이 없습니다.")
        return

    original_epsilon = agent.epsilon
    agent.epsilon = 0.01 # 테스트 시 탐험 최소화

    actual_num_tests = min(num_tests, len(test_puzzles))
    
    for i in range(actual_num_tests):
        puzzle_to_test = np.array(test_puzzles[i]).copy()
        print(f"\n--- 테스트 {i+1}/{actual_num_tests} ---")
        print("테스트 퍼즐:")
        temp_env_render = SudokuEnv(grid_size=env.grid_size) # 렌더링용 임시 환경
        temp_env_render.reset(puzzle=puzzle_to_test.copy())
        temp_env_render.render()
        
        _ = env.reset(puzzle=puzzle_to_test.copy())
        current_agent_state = env._get_current_agent_state()
        done = False
        
        # 최대 스텝은 (빈칸의 수) + 약간의 여유분. 빈칸은 최대 grid_size^2.
        # 에이전트가 최적으로 행동한다면 빈칸의 수만큼 스텝이 필요.
        max_test_steps = env.grid_size * env.grid_size + 5 

        for step_num in range(max_test_steps):
            if current_agent_state is None or done: # 모든 셀이 채워졌거나, 이미 종료된 경우
                break
            
            if not current_agent_state[1]: # 가능한 숫자가 없는 경우
                print("테스트 중 막힌 상태 (가능한 숫자 없음)")
                break

            action_chosen_number = agent.choose_action(current_agent_state) # 상태를 기반으로 숫자 선택
            
            if action_chosen_number is None: # 에이전트가 행동을 선택 못한 경우
                print("테스트 중 에이전트가 행동을 선택하지 못함.")
                break
            
            next_agent_state, _, done, info = env.step(action_chosen_number)
            
            print(f"테스트 스텝 {step_num+1}: {current_agent_state[0]}에 {action_chosen_number} 놓음. 정보: {info}")
            env.render() # 각 스텝 후 보드 상태 출력

            current_agent_state = next_agent_state
            
            if done:
                break

        if env.is_solved():
            solved_count += 1
            print("해결 성공!")
        else:
            print("해결 실패.")
            print("최종 보드 상태:")
            env.render() # 실패 시 최종 상태 한 번 더 보여주기 (이미 render 했지만 명시적)
            
    print(f"\n테스트 완료. 총 {actual_num_tests}번 시도 중 {solved_count}번 해결 (성공률: {solved_count/actual_num_tests:.3f})")
    agent.epsilon = original_epsilon

if __name__ == '__main__':
    GRID_SIZE = 4
    EPISODES = 1000000 # 학습 에피소드 수 (테스트를 위해 줄임, 원래값 1000000)
    MAX_STEPS_PER_EPISODE = GRID_SIZE * GRID_SIZE + 5 # 에피소드당 최대 스텝 수 (4x4 경우 16칸 + 여유)
    EPSILON_DECAY_STEPS = EPISODES * 0.8 
    USE_RANDOM_PUZZLES_FOR_TRAINING = True

    sudoku_env = SudokuEnv(grid_size=GRID_SIZE)
    
    # QLearningAgent 초기화 시 env 인스턴스 제거 및 action_space_size 제거
    q_agent = QLearningAgent(
        # env=sudoku_env, # 이 부분 제거
        learning_rate=0.001,
        discount_factor=0.99, 
        epsilon_start=1.0, 
        epsilon_end=0.01, 
        epsilon_decay_steps=EPSILON_DECAY_STEPS
    )

    # 에이전트 학습
    rewards, trained_agent = train_agent(
        sudoku_env, 
        q_agent, 
        episodes=EPISODES, 
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        use_random_puzzles=USE_RANDOM_PUZZLES_FOR_TRAINING
    )

    # 학습 결과 시각화
    plot_rewards(rewards, title=f'{GRID_SIZE}x{GRID_SIZE} Sudoku Q-Learning Rewards')

    # 학습된 에이전트 테스트
    # EXAMPLE_PUZZLES_4X4에서 학습에 사용되지 않았거나 다른 퍼즐들로 테스트
    test_puzzles_list = [
        p for p in EXAMPLE_PUZZLES_4X4[:5] # 처음 5개 퍼즐로 테스트 (또는 다른 퍼즐 선택)
    ]
    if not USE_RANDOM_PUZZLES_FOR_TRAINING and EXAMPLE_PUZZLES_4X4: # 고정퍼즐로 학습했다면 다른 퍼즐로 테스트
        test_puzzles_list = [p for p in EXAMPLE_PUZZLES_4X4[1:6] if EXAMPLE_PUZZLES_4X4]

    test_agent(sudoku_env, trained_agent, test_puzzles=test_puzzles_list, num_tests=5)

    # Q-테이블 크기 확인 (참고용)
    print(f"\nQ-테이블의 상태 수: {len(trained_agent.q_table)}")
    q_table_sample = list(trained_agent.q_table.items())[:2]
    if q_table_sample:
        print("Q-테이블 샘플 (키: 상태((row, col), (가능한 숫자들)), 값: {숫자: Q값}):")
        for state, actions_q_values in q_table_sample:
            # 상태 표현이 ((r,c), (num1, num2,...)) 이므로 그대로 출력
            print(f"  상태 {state}: {actions_q_values}") 
