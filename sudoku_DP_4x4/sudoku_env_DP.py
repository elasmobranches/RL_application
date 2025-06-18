import numpy as np
import itertools
from collections import defaultdict

# 4x4 스도쿠 환경을 구현한 클래스
class SudokuEnv:
    def __init__(self, board=None):
        self.board_size = 4  # 4x4 스도쿠
        self.box_size = 2    # 2x2 박스
        # 보드가 주어지지 않으면 빈 보드 생성
        if board is None: 
            self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        else:
            self.board = board.copy()
        self.initial_board = self.board.copy()
        self._cache = defaultdict(bool)  # 상태 검증 캐시

    # 빈 칸의 위치를 반환
    def get_empty_cells(self):
        return [(r, c) for r, c in itertools.product(range(self.board_size), range(self.board_size)) if self.board[r, c] == 0]

    # 주어진 위치에 숫자를 놓을 수 있는지 검사
    def is_valid(self, row, col, num, board=None):
        """
        주어진 위치에 숫자를 놓을 수 있는지 검사합니다.
        board가 주어지면 해당 보드에 대해 검사하고, 아니면 self.board에 대해 검사합니다.
        """
        if board is None:
            board = self.board
            cache_key = (row, col, num)
            if cache_key in self._cache:
                return self._cache[cache_key]
        # 행, 열, 박스에 같은 숫자가 있으면 False 반환
        if num in board[row, :]:
            if board is self.board:
                self._cache[cache_key] = False
            return False
        if num in board[:, col]:
            if board is self.board:
                self._cache[cache_key] = False
            return False
        box_size = int(np.sqrt(board.shape[0]))
        start_row, start_col = box_size * (row // box_size), box_size * (col // box_size)
        if num in board[start_row:start_row + box_size, start_col:start_col + box_size]:
            if board is self.board:
                self._cache[cache_key] = False
            return False
        if board is self.board:
            self._cache[cache_key] = True
        return True

    # 현재 상태에서 가능한 모든 행동(빈 칸에 넣을 수 있는 숫자)을 반환 (0,0부터 3,3까지 차례대로)
    def get_possible_actions(self, state):
        board = self.get_board_from_state(state)
        empty_cells = self.get_empty_cells_from_board(board)
        if not empty_cells:
            return []
        actions = []
        for r, c in empty_cells:
            for num in range(1, self.board_size + 1):
                if self.is_valid(r, c, num, board):
                    actions.append(((r, c), num))
        return actions

    # 주어진 행동을 수행하고 다음 상태, 보상, 종료 여부를 반환
    def step(self, action):
        row, col = action[0]
        num = action[1]
        # 이미 채워진 칸에 놓으려고 하면 즉시 종료
        if self.board[row, col] != 0:
            return self.board.copy(), -1.0, True
        # 숫자를 놓기 전에 유효성 검사
        if not self.is_valid(row, col, num):
            return self.board.copy(), -1.0, True
        # 숫자를 놓고 보상 계산
        self.board[row, col] = num
        self._clear_related_cache(row, col)
        reward = self._calculate_reward(row, col)
        # 퍼즐 완성 여부 확인
        done = len(self.get_empty_cells()) == 0
        if done:
            if SudokuEnv.is_fully_solved(self.board):
                reward += 10.0  # 퍼즐 완성 보상
            else:
                reward = -10.0  # 잘못된 퍼즐 완성 페널티
        return self.board.copy(), reward, done

    # 주어진 위치에 숫자를 놓았을 때의 보상을 계산
    def _calculate_reward(self, row, col):
        reward = 0.0
        # 행, 열, 박스가 완성되면 보상, 중복 있으면 페널티
        if 0 not in self.board[row, :]:
            if len(set(self.board[row, :])) == self.board_size:
                reward += 1.0
            else:
                return -1.0
        if 0 not in self.board[:, col]:
            if len(set(self.board[:, col])) == self.board_size:
                reward += 1.0
            else:
                return -1.0
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        box = self.board[start_row:start_row + self.box_size, start_col:start_col + self.box_size]
        if 0 not in box:
            if len(set(box.flatten())) == self.board_size:
                reward += 1.0
            else:
                return -1.0
        return reward

    # 스도쿠 퍼즐이 완성되었는지 검사
    @staticmethod
    def is_fully_solved(board):
        size = board.shape[0]
        box_size = int(np.sqrt(size))
        target_sum = sum(range(1, size + 1))
        for i in range(size):
            if sum(board[i, :]) != target_sum or len(set(board[i, :])) != size:
                return False
            if sum(board[:, i]) != target_sum or len(set(board[:, i])) != size:
                return False
        for i in range(0, size, box_size):
            for j in range(0, size, box_size):
                box = board[i:i+box_size, j:j+box_size].flatten()
                if sum(box) != target_sum or len(set(box)) != size:
                    return False
        return True

    # 환경을 초기 상태로 리셋
    def reset(self):
        self.board = self.initial_board.copy()
        self._cache.clear()
        return self.board.copy()
    
    # 현재 상태를 튜플로 반환
    def get_state(self):
        return tuple(self.board.flatten())

    # 상태 튜플을 보드로 변환
    @staticmethod
    def get_board_from_state(state):
        size = int(np.sqrt(len(state)))
        return np.array(state).reshape((size, size))

    # 주어진 보드에서 빈 칸의 위치 반환
    def get_empty_cells_from_board(self, board):
        size = board.shape[0]
        return [(r, c) for r, c in itertools.product(range(size), range(size)) if board[r, c] == 0]

    # 특정 위치와 관련된 캐시 항목만 제거
    def _clear_related_cache(self, row, col):
        for c in range(self.board_size):
            for n in range(1, self.board_size + 1):
                self._cache.pop((row, c, n), None)
        for r in range(self.board_size):
            for n in range(1, self.board_size + 1):
                self._cache.pop((r, col, n), None)
        start_row = self.box_size * (row // self.box_size)
        start_col = self.box_size * (col // self.box_size)
        for r in range(start_row, start_row + self.box_size):
            for c in range(start_col, start_col + self.box_size):
                for n in range(1, self.board_size + 1):
                    self._cache.pop((r, c, n), None) 