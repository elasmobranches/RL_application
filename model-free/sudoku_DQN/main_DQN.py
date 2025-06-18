import numpy as np
from sudoku_env_DQN import SudokuEnv
from DQN_agent_cnn import DQNAgent # 새로 만든 sudoku_DQN_agent.py에서 임포트
from utils import plot_rewards, get_random_puzzle_4x4, EXAMPLE_PUZZLES_4X4, plot_losses

def train_dqn_agent(env, agent, episodes=10000, max_steps_per_episode=100, use_random_puzzles=True):
    episode_rewards = []
    all_losses = [] # 모든 스텝의 손실을 기록 (또는 에피소드별 평균 손실)
    solved_count = 0
    total_steps = 0

    print(f"DQN 학습 시작: 총 {episodes} 에피소드, 에피소드 당 최대 스텝: {max_steps_per_episode}")
    if use_random_puzzles:
        print("매 에피소드마다 무작위 퍼즐을 사용합니다.")
    else:
        fixed_puzzle = np.array(EXAMPLE_PUZZLES_4X4[0]) if EXAMPLE_PUZZLES_4X4 else None
        if fixed_puzzle is not None:
            print(f"고정된 퍼즐로 학습합니다: {fixed_puzzle.tolist()}")
        else:
            print("고정 퍼즐이 없어 빈 보드로 학습합니다. (무작위 퍼즐 사용 권장)")
            use_random_puzzles = True # Fallback

    for episode in range(episodes):
        if use_random_puzzles:
            puzzle = get_random_puzzle_4x4()
            current_board_state = env.reset(puzzle=puzzle.copy() if puzzle is not None else None)
        else:
            current_board_state = env.reset(puzzle=fixed_puzzle.copy() if fixed_puzzle is not None else None)
        
        total_reward_for_episode = 0
        current_episode_losses = [] # 현재 에피소드의 스텝별 손실 기록용
        
        for step_count in range(max_steps_per_episode):
            total_steps += 1
            
            # 현재 빈칸 정보 및 가능한 숫자(행동 마스크용) 가져오기
            current_cell_coord, possible_numbers_tuple = env.get_current_empty_cell_and_possible_numbers()

            if current_cell_coord is None: # 모든 칸이 채워짐 (풀렸거나, 꽉 찼지만 못 품)
                break 

            action_mask = np.zeros(env.grid_size, dtype=bool)
            if possible_numbers_tuple:
                for num in possible_numbers_tuple: # possible_numbers_tuple은 1-indexed
                    if 1 <= num <= env.grid_size:
                        action_mask[num-1] = True # 0-indexed 마스크
            
            if not np.any(action_mask): # 가능한 행동이 없음 (막힌 상태)
                # 이론적으로 env.step에서 페널티 받고 done 처리되거나, 
                # 또는 이전에 current_cell_coord is None으로 루프 탈출
                # main 루프에서는 이 상황이 오면 다음 스텝 진행이 어려움
                break

            chosen_number_action = agent.choose_action(current_board_state, action_mask) # 숫자를 반환 (1-indexed)

            if chosen_number_action is None: # 에이전트가 행동을 선택하지 못한 경우 (예: 마스크가 다 False)
                break

            next_board_state, reward, done, info = env.step(chosen_number_action)
            
            agent.store_transition(current_board_state, chosen_number_action, reward, next_board_state, done)
            
            loss = None # loss 변수 초기화
            if len(agent.replay_buffer) > agent.batch_size :
                 loss = agent.learn() # learn 메소드가 손실 값을 반환
                 if loss is not None:
                     current_episode_losses.append(loss)
            
            current_board_state = next_board_state
            total_reward_for_episode += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward_for_episode)
        if current_episode_losses: # 현재 에피소드에서 학습이 일어났다면
            all_losses.append(np.mean(current_episode_losses)) # 에피소드 평균 손실 저장

        if env.is_solved():
            solved_count += 1
            
        if (episode + 1) % (episodes // 20 if episodes >= 20 else 1) == 0:
            avg_reward_recent = np.mean(episode_rewards[-(episodes // 20 if episodes >= 20 else 1):])
            avg_loss_recent = np.mean(all_losses[-(episodes // 20 if episodes >= 20 else 1):]) if all_losses else 0 # 최근 평균 손실
            current_solve_rate = solved_count / (episode + 1)
            print(f"Episode {episode + 1}/{episodes} - 최근 평균 보상: {avg_reward_recent:.2f}, "
                  f"최근 평균 손실: {avg_loss_recent:.4f}, Epsilon: {agent.epsilon:.3f}, "
                  f"누적 성공률: {current_solve_rate:.3f}, Buffer: {len(agent.replay_buffer)}, Total Steps: {total_steps}")
            # env.render() # 필요시 현재 보드 상태 출력
                            
    print(f"학습 완료. 총 {episodes} 에피소드 중 {solved_count}번 해결 (성공률: {solved_count/episodes:.3f})")
    print("학습 과정 그래프가 표시되었거나 파일로 저장되었습니다.")
    print("Press Enter to save the model and proceed to testing...")
    input() # 사용자가 그래프를 확인하고 엔터를 누를 때까지 대기
    return episode_rewards, all_losses, agent

def test_dqn_agent(env, agent, test_puzzles, num_tests=5):
    print("\nDQN 에이전트 테스트 시작...")
    solved_count = 0
    
    if not test_puzzles:
        print("테스트할 퍼즐이 없습니다.")
        return

    original_epsilon = agent.epsilon # 테스트 전 엡실론 저장
    agent.epsilon = 0.00 # 테스트 시 탐험 없음 (greedy)
    actual_num_tests = min(num_tests, len(test_puzzles))
    
    for i in range(actual_num_tests):
        puzzle_to_test = np.array(test_puzzles[i]).copy()
        print(f"\n--- 테스트 {i+1}/{actual_num_tests} ---")
        print("테스트 퍼즐:")
        temp_env_render = SudokuEnv(grid_size=env.grid_size)
        temp_env_render.reset(puzzle=puzzle_to_test.copy())
        temp_env_render.render()
        
        current_board_state = env.reset(puzzle=puzzle_to_test.copy())
        done = False
        
        for step_num in range(env.grid_size * env.grid_size + 5):
            current_cell_coord, possible_numbers_tuple = env.get_current_empty_cell_and_possible_numbers()
            if current_cell_coord is None or done:
                break
            
            action_mask = np.zeros(env.grid_size, dtype=bool)
            if possible_numbers_tuple:
                for num in possible_numbers_tuple:
                     if 1 <= num <= env.grid_size:
                        action_mask[num-1] = True
            
            if not np.any(action_mask):
                print("테스트 중 막힌 상태 (가능한 숫자 없음)")
                break

            chosen_number_action = agent.choose_action(current_board_state, action_mask)
            
            if chosen_number_action is None:
                print("테스트 중 에이전트가 행동을 선택하지 못함.")
                break
            
            next_board_state, _, done, info = env.step(chosen_number_action)
            
            # print(f"테스트 스텝 {step_num+1}: {current_cell_coord}에 {chosen_number_action} 놓음. 정보: {info}")
            # env.render() # 너무 많은 로그를 피하기 위해 테스트 중 렌더링은 선택적으로

            current_board_state = next_board_state
            if done:
                break

        if env.is_solved():
            solved_count += 1
            print("해결 성공!")
            env.render() # 성공 시 최종 보드 출력
        else:
            print("해결 실패.")
            print("최종 보드 상태:")
            env.render()
            
    print(f"\n테스트 완료. 총 {actual_num_tests}번 시도 중 {solved_count}번 해결 (성공률: {solved_count/actual_num_tests:.3f})")
    agent.epsilon = original_epsilon # 원래 엡실론으로 복원


if __name__ == '__main__':
    GRID_SIZE = 4  # 4x4 스도쿠
    EPISODES = 50000 # DQN 학습 에피소드 수 (Q-러닝보다 적게 시작해볼 수 있음)
    MAX_STEPS_PER_EPISODE = GRID_SIZE * GRID_SIZE + 5 
    
    # DQN 하이퍼파라미터
    LEARNING_RATE = 0.0005 # 일반적인 DQN 학습률
    DISCOUNT_FACTOR = 0.9
    EPSILON_START = 1.0
    EPSILON_END = 0.05 # Q-러닝보다 조금 높게 설정하여 탐험 유지 가능
    EPSILON_DECAY_STEPS = EPISODES * 0.8 # 예시: 80% 지점에서 최소 엡실론 도달
    REPLAY_BUFFER_CAPACITY = 50000
    BATCH_SIZE = 128       # 일반적인 배치 크기
    TARGET_UPDATE_FREQ = 200 # 타겟 네트워크 업데이트 빈도 (스텝 기준)
    USE_DOUBLE_DQN = True

    USE_RANDOM_PUZZLES_FOR_TRAINING = True

    sudoku_env = SudokuEnv(grid_size=GRID_SIZE)
    dqn_agent = DQNAgent(
        grid_size=GRID_SIZE,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_steps=EPSILON_DECAY_STEPS,
        replay_buffer_capacity=REPLAY_BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        use_double_dqn=USE_DOUBLE_DQN
    )



    # 에이전트 학습
    rewards_history, losses_history, trained_dqn_agent = train_dqn_agent(
        sudoku_env, 
        dqn_agent, 
        episodes=EPISODES, 
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        use_random_puzzles=USE_RANDOM_PUZZLES_FOR_TRAINING
    )

    # 학습 결과 시각화
    plot_rewards(rewards_history, title=f'{GRID_SIZE}x{GRID_SIZE} Sudoku DQN Rewards (DoubleDQN: {USE_DOUBLE_DQN})')
    if losses_history: # 손실 기록이 있을 경우에만 플롯
        plot_losses(losses_history, title=f'{GRID_SIZE}x{GRID_SIZE} Sudoku DQN Average Loss per Episode')

    # 학습된 에이전트 저장 (필요시)
    trained_dqn_agent.save_model(f"sudoku_dqn_agent_{GRID_SIZE}x{GRID_SIZE}.pth")

    # 학습된 에이전트 테스트
    test_puzzles_list = [p for p in EXAMPLE_PUZZLES_4X4[:5]]
    if not USE_RANDOM_PUZZLES_FOR_TRAINING and EXAMPLE_PUZZLES_4X4:
        test_puzzles_list = [p for p in EXAMPLE_PUZZLES_4X4[1:6] if EXAMPLE_PUZZLES_4X4]
    
    # 새로 에이전트를 로드하여 테스트하려면:
    test_agent = DQNAgent(grid_size=GRID_SIZE) 
    test_agent.load_model(f"sudoku_dqn_agent_{GRID_SIZE}x{GRID_SIZE}.pth")
    test_dqn_agent(sudoku_env, test_agent, test_puzzles=test_puzzles_list, num_tests=5)
    
    test_dqn_agent(sudoku_env, trained_dqn_agent, test_puzzles=test_puzzles_list, num_tests=5) 