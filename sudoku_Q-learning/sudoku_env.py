import numpy as np
import random

class SudokuEnv:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.sqrt_grid_size = int(np.sqrt(grid_size))
        self.board = np.zeros((grid_size, grid_size), dtype=int)
        self.empty_cells_coords = [] # (row, col) 튜플 저장
        # self._update_empty_cells_coords() # reset에서 호출

    def reset(self, puzzle=None):
        if puzzle is not None:
            self.board = np.array(puzzle).reshape((self.grid_size, self.grid_size))
        else:
            self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self._update_empty_cells_coords()
        # train_agent에서 _get_current_agent_state()를 호출하므로, 여기서는 보드 상태만 반환하거나,
        # 이전처럼 flatten된 배열을 반환할 수 있습니다. main.py의 reset 호출 부분을 봐야 함.
        # 일단 이전 호환성을 위해 flatten된 배열 반환
        return self.board.flatten()

    def _update_empty_cells_coords(self):
        self.empty_cells_coords = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == 0:
                    self.empty_cells_coords.append((r, c))
        self.empty_cells_coords.sort() # 일관된 순서로 처리 (예: top-left 우선)

    def _get_possible_numbers_for_cell(self, r, c):
        if self.board[r,c] != 0: # 이미 채워진 셀에 대해서는 호출되면 안됨
             return tuple()

        possible_nums = []
        for num_candidate in range(1, self.grid_size + 1):
            if self.is_valid(r, c, num_candidate):
                possible_nums.append(num_candidate)
        return tuple(sorted(possible_nums))

    def _get_current_agent_state(self):
        if not self.empty_cells_coords:
            return None # 모든 셀이 채워짐
        
        # 항상 첫 번째 빈칸을 기준으로 상태 정의
        r, c = self.empty_cells_coords[0]
        possible_nums = self._get_possible_numbers_for_cell(r, c)
        return ((r, c), possible_nums)

    def is_valid(self, row, col, num):
        # 행 검사 (자기 자신 위치는 제외하지 않음 - is_valid는 빈칸에 놓을 때를 가정)
        for c_idx in range(self.grid_size):
            if self.board[row, c_idx] == num:
                return False
        # 열 검사
        for r_idx in range(self.grid_size):
            if self.board[r_idx, col] == num:
                return False
        # 서브그리드 검사
        start_row, start_col = self.sqrt_grid_size * (row // self.sqrt_grid_size), self.sqrt_grid_size * (col // self.sqrt_grid_size)
        for r_offset in range(self.sqrt_grid_size):
            for c_offset in range(self.sqrt_grid_size):
                if self.board[start_row + r_offset, start_col + c_offset] == num:
                    return False
        return True

    def get_state(self): # 기존 get_state는 flatten된 보드 전체를 반환했었음. 유지 또는 변경.
        # 에이전트는 _get_current_agent_state()를 사용하므로, 이 함수는 이전 로직 유지를 위해 남겨둘 수 있음.
        return self.board.flatten()

    def is_solved(self):
        if self.empty_cells_coords: # 빈칸이 남아있으면 해결된 것이 아님
            return False
        return self._check_all_rules() # 모든 칸이 채워졌을 때만 규칙 검사

    def _check_all_rules(self):
        # 이 함수는 self.empty_cells_coords가 비어 있을 때 호출되므로,
        # 보드에 0 (빈칸)이 없다고 가정할 수 있습니다.
        # 모든 셀은 1에서 self.grid_size 사이의 숫자로 채워져 있어야 합니다.

        # 행 검사
        for r_idx in range(self.grid_size):
            # 해당 행의 숫자들로 set을 만들고, 그 크기가 grid_size와 다른 경우 중복이 있음을 의미
            if len(set(self.board[r_idx, :])) != self.grid_size:
                return False

        # 열 검사
        for c_idx in range(self.grid_size):
            # 해당 열의 숫자들로 set을 만들고, 그 크기가 grid_size와 다른 경우 중복이 있음을 의미
            if len(set(self.board[:, c_idx])) != self.grid_size:
                return False

        # 서브그리드 검사
        for r_box_start in range(0, self.grid_size, self.sqrt_grid_size):
            for c_box_start in range(0, self.grid_size, self.sqrt_grid_size):
                subgrid = self.board[r_box_start : r_box_start + self.sqrt_grid_size, \
                                     c_box_start : c_box_start + self.sqrt_grid_size]
                # 해당 서브그리드의 숫자들로 set을 만들고 (flatten 후), 
                # 그 크기가 grid_size와 다른 경우 중복이 있음을 의미
                if len(set(subgrid.flatten())) != self.grid_size:
                    return False
        
        return True

    def step(self, chosen_number):
        info = {}
        done = False
        
        current_agent_state_before_action = self._get_current_agent_state()

        if not self.empty_cells_coords:
            info['error'] = "Step called on a board with no empty cells"
            return current_agent_state_before_action, 0, True, info

        r, c = self.empty_cells_coords[0] 

        possible_nums = self._get_possible_numbers_for_cell(r, c)
        if chosen_number not in possible_nums:
            reward = -100 
            done = True
            info['error'] = f"Invalid number {chosen_number} chosen for cell ({r},{c}). Possible: {possible_nums}"
            return current_agent_state_before_action, reward, done, info

        self.board[r, c] = chosen_number
        self._update_empty_cells_coords()

        reward = 0
        if self.is_solved():
            reward = 1000 
            done = True
        elif not self.empty_cells_coords: 
            reward = -1000 
            done = True
            info['error'] = "Board full but not solved"
        else:
            reward = -0.1 

        next_agent_state = self._get_current_agent_state()

        # 제안된 "막힌 상태" 검사 로직 추가
        if not done and next_agent_state is not None and not next_agent_state[1]: # 다음 상태가 존재하지만 가능한 숫자가 없음
            reward = -1000 # 막힌 상태에 대한 큰 페널티
            done = True
            info['error'] = "Agent reached a dead-end (no possible numbers for next cell)"
            # 이 경우 next_agent_state 자체는 유효한 ((r,c), ()) 형태일 수 있으므로 그대로 반환
            # 또는, 이 상태에 대한 Q값 학습을 원치 않는다면 다른 처리가 필요할 수 있으나, 현재는 이대로 둠.

        return next_agent_state, reward, done, info

    def render(self):
        for r in range(self.grid_size):
            if r % self.sqrt_grid_size == 0 and r != 0:
                print("-" * (self.grid_size * 2 + self.sqrt_grid_size - 1))
            row_str = []
            for c in range(self.grid_size):
                if c % self.sqrt_grid_size == 0 and c != 0:
                    row_str.append("|")
                row_str.append(str(self.board[r, c]) if self.board[r, c] != 0 else '.')
            print(" ".join(row_str))
        print()

if __name__ == '__main__':
    # SudokuEnv 테스트 코드
    env = SudokuEnv(grid_size=4)
    print("초기 상태:")
    env.render()
    
    example_puzzle = [
        [1, 0, 3, 0],
        [0, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 0, 4]
    ]
    env.reset(puzzle=example_puzzle)
    print("예시 퍼즐 상태:")
    env.render()

    print(f"초기 빈칸: {env.empty_cells_coords}")
    # 예상 초기 빈칸: [(0, 1), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2), (2, 3), (3, 0), (3, 1), (3, 2)]
    print(f"초기 상태 (flattened): {env.get_state()}")

    # 첫 번째 행동:
    # 현재 빈칸 중 첫 번째는 env.empty_cells_coords[0] (예: (0,1))
    # (0,1)에 가능한 숫자: env._get_possible_numbers_for_cell(0,1) -> (2, 4)
    # (0,1)에 4를 놓는 시도 (유효)
    if env.empty_cells_coords:
        target_cell_1 = env.empty_cells_coords[0]
        chosen_num_1 = 4 
        print(f"\n첫 번째 행동: {target_cell_1}에 {chosen_num_1} 놓기 시도")
        possible_before_step1 = env._get_possible_numbers_for_cell(*target_cell_1)
        print(f"놓기 전 {target_cell_1}에 가능한 숫자: {possible_before_step1}")

        next_state, reward, done, info = env.step(chosen_num_1)
        print(f"행동 결과: 보상: {reward}, 종료: {done}, 정보: {info}")
        env.render()
    else:
        print("\n첫 번째 행동을 위한 빈 칸이 없습니다.")

    # 두 번째 행동:
    # 이전 행동으로 (0,1)이 채워졌다고 가정. 다음 빈칸은 env.empty_cells_coords[0] (예: (0,3))
    # (0,3)에 4를 놓는 시도 (유효하지 않음 - 행 (0,*)에 1, (이전 단계에서 놓은)4, 3이 있으므로 4는 중복)
    # step 함수 내 유효성 검사에 의해 페널티 및 done=True 예상
    if env.empty_cells_coords and not done: # 아직 게임이 끝나지 않았고 빈 칸이 있다면
        target_cell_2 = env.empty_cells_coords[0] # 예: (0,3)
        chosen_num_2 = 4 # 유효하지 않을 수 있는 숫자
        print(f"\n두 번째 행동: {target_cell_2}에 {chosen_num_2} 놓기 시도")
        possible_before_step2 = env._get_possible_numbers_for_cell(*target_cell_2)
        print(f"놓기 전 {target_cell_2}에 가능한 숫자: {possible_before_step2}")
        
        next_state, reward, done, info = env.step(chosen_num_2)
        print(f"행동 결과: 보상: {reward}, 종료: {done}, 정보: {info}")
        env.render()
    elif done:
        print("\n첫 번째 행동으로 게임이 종료되어 두 번째 행동을 수행할 수 없습니다.")
    else:
        print("\n두 번째 행동을 위한 빈 칸이 없습니다.")

    # 세 번째 행동: 유효한 숫자 놓기 (만약 게임이 계속된다면)
    if env.empty_cells_coords and not done:
        target_cell_3 = env.empty_cells_coords[0]
        possible_nums_3 = env._get_possible_numbers_for_cell(*target_cell_3)
        if possible_nums_3:
            chosen_num_3 = possible_nums_3[0] # 가능한 숫자 중 첫 번째 선택
            print(f"\n세 번째 행동: {target_cell_3}에 {chosen_num_3} (유효한 숫자) 놓기 시도")
            next_state, reward, done, info = env.step(chosen_num_3)
            print(f"행동 결과: 보상: {reward}, 종료: {done}, 정보: {info}")
            env.render()
        else:
            print(f"\n{target_cell_3}에 놓을 수 있는 유효한 숫자가 없습니다.")
    elif done and not env.empty_cells_coords : # 모든 칸이 채워졌으나 해결되지 않은 경우도 고려
         print(f"\n두 번째 행동으로 게임이 종료되었습니다. 해결 여부: {env.is_solved()}")
    elif done:
         print("\n이전 행동으로 게임이 종료되어 세 번째 행동을 수행할 수 없습니다.")


    print(f"\n테스트 후 빈칸 수: {len(env.empty_cells_coords)}")
    print(f"스도쿠 해결 여부: {env.is_solved()}") 