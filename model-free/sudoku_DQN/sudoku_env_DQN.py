import numpy as np
import random

class SudokuEnv:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.sqrt_grid_size = int(np.sqrt(grid_size))
        self.board = np.zeros((grid_size, grid_size), dtype=int)
        self.empty_cells_coords = [] 
        # self._update_empty_cells_coords() # reset에서 호출

    def reset(self, puzzle=None):
        if puzzle is not None:
            self.board = np.array(puzzle).reshape((self.grid_size, self.grid_size))
        else:
            self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        self._update_empty_cells_coords()
        return self.board.copy() # DQN 입력을 위해 2D 보드 상태 반환

    def _update_empty_cells_coords(self):
        self.empty_cells_coords = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.board[r, c] == 0:
                    self.empty_cells_coords.append((r, c))
        self.empty_cells_coords.sort() 

    def _get_possible_numbers_for_cell(self, r, c):
        if self.board[r,c] != 0: 
             return tuple()

        possible_nums = []
        for num_candidate in range(1, self.grid_size + 1):
            if self.is_valid(r, c, num_candidate):
                possible_nums.append(num_candidate)
        return tuple(sorted(possible_nums))

    def get_current_empty_cell_and_possible_numbers(self):
        """DQN 에이전트가 사용할 현재 빈칸 정보와 가능한 숫자들을 반환합니다."""
        if not self.empty_cells_coords:
            return None, tuple() # (cell_coord, possible_nums)
        
        r, c = self.empty_cells_coords[0]
        possible_nums = self._get_possible_numbers_for_cell(r, c)
        return (r, c), possible_nums

    def is_valid(self, row, col, num):
        # 행 검사
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

    def get_state(self): # 이전 버전과의 호환성 또는 다른 용도로 남겨둘 수 있음
        return self.board.flatten()

    def is_solved(self):
        if self.empty_cells_coords: 
            return False
        return self._check_all_rules()

    def _check_all_rules(self):
        # 행 검사
        for r_idx in range(self.grid_size):
            if len(set(self.board[r_idx, :])) != self.grid_size:
                return False
        # 열 검사
        for c_idx in range(self.grid_size):
            if len(set(self.board[:, c_idx])) != self.grid_size:
                return False
        # 서브그리드 검사
        for r_box_start in range(0, self.grid_size, self.sqrt_grid_size):
            for c_box_start in range(0, self.grid_size, self.sqrt_grid_size):
                subgrid = self.board[r_box_start : r_box_start + self.sqrt_grid_size, \
                                     c_box_start : c_box_start + self.sqrt_grid_size]
                if len(set(subgrid.flatten())) != self.grid_size:
                    return False
        return True
    
    def step(self, chosen_number): # chosen_number는 1부터 grid_size까지의 값
        info = {}
        done = False
        
        if not self.empty_cells_coords:
            info['error'] = "Step called on a board with no empty cells"
            # 다음 상태는 현재 상태 그대로, 보상 0, 종료 True
            return self.board.copy(), 0, True, info

        r, c = self.empty_cells_coords[0] 

        possible_nums = self._get_possible_numbers_for_cell(r, c)
        
        # chosen_number는 1-indexed, possible_nums도 1-indexed
        if chosen_number not in possible_nums:
            reward = -10
            done = True # 잘못된 행동은 에피소드 종료
            info['error'] = f"Invalid number {chosen_number} chosen for cell ({r},{c}). Possible: {possible_nums}"
            # 보드는 변경하지 않고, 이전 상태와 페널티, 종료 상태 반환
            return self.board.copy(), reward, done, info

        self.board[r, c] = chosen_number
        self._update_empty_cells_coords()

        reward = 0
        if self.is_solved():
            reward = 100 
            done = True
        elif not self.empty_cells_coords: 
            reward = -100
            done = True
            info['error'] = "Board full but not solved"
        else:
            reward = 1 # 한 칸을 올바르게 채웠을 때 긍정 보상 (새로 추가)
            # 다음 빈칸에 놓을 수 있는 숫자가 없는지 확인 (막힌 상태)
            _ , next_possible_nums = self.get_current_empty_cell_and_possible_numbers()
            if not next_possible_nums and not self.empty_cells_coords: # 모든 칸이 안채워졌는데 다음 수가 없는 경우 (실제로는 이전에 empty_cells_coords로 걸러짐)
                 pass # 이 경우는 거의 발생 안함
            elif not next_possible_nums and self.empty_cells_coords : # 아직 빈칸이 있는데 다음 수가 없는경우
                reward = -500 # 막힌 상태에 대한 큰 페널티
                done = True
                info['error'] = "Agent reached a dead-end (no possible numbers for next cell)"
            #else: # 게임 진행 중
                #reward = -0.1 
        
        return self.board.copy(), reward, done, info

    def render(self):
        for r_idx in range(self.grid_size):
            if r_idx % self.sqrt_grid_size == 0 and r_idx != 0:
                print("-" * (self.grid_size * 2 + self.sqrt_grid_size - 1))
            row_str = []
            for c_idx in range(self.grid_size):
                if c_idx % self.sqrt_grid_size == 0 and c_idx != 0:
                    row_str.append("|")
                row_str.append(str(self.board[r_idx, c_idx]) if self.board[r_idx, c_idx] != 0 else '.')
            print(" ".join(row_str))
        print()

if __name__ == '__main__':
    env = SudokuEnv(grid_size=4)
    print("초기 상태:")
    env.render()
    
    example_puzzle = [
        [1, 0, 3, 0], [0, 0, 0, 2],
        [0, 1, 0, 0], [0, 0, 0, 4]
    ]
    initial_board = env.reset(puzzle=example_puzzle)
    print("예시 퍼즐 상태:")
    env.render()
    print(f"Reset 반환 (보드 상태):\n{initial_board}")

    cell_coord, p_nums = env.get_current_empty_cell_and_possible_numbers()
    print(f"현재 빈칸: {cell_coord}, 가능한 숫자: {p_nums}")

    if p_nums:
        action = p_nums[0] # 가능한 숫자 중 첫번째 선택
        print(f"\n선택한 행동 (숫자): {action}을 {cell_coord}에 놓기")
        next_board, reward, done, info = env.step(action)
        
        print("행동 후 상태:")
        env.render()
        print(f"다음 상태 (보드):\n{next_board}")
        print(f"보상: {reward}, 종료: {done}, 정보: {info}")

        if not done:
            cell_coord2, p_nums2 = env.get_current_empty_cell_and_possible_numbers()
            print(f"\n다음 빈칸: {cell_coord2}, 가능한 숫자: {p_nums2}")

