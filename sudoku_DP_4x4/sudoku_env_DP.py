import numpy as np
import itertools
from collections import defaultdict

class SudokuEnv:
    def __init__(self, board=None):
        self.board_size = 4
        self.box_size = 2
        # 그리드 없을시 빈 보드 생성(굳이 필요하진 않아보임)
        if board is None: 
            self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        else:
            self.board = board.copy()
        self.initial_board = self.board.copy()
        self._cache = defaultdict(bool)  # 상태 검증 캐시

    def get_empty_cells(self):
        return [(r, c) for r, c in itertools.product(range(self.board_size), range(self.board_size)) if self.board[r, c] == 0]

    def is_valid(self, row, col, num, board=None):
        """
        주어진 위치에 숫자를 놓을 수 있는지 검사합니다.
        board가 주어지면 해당 보드에 대해 검사하고, 아니면 self.board에 대해 검사합니다.
        """
        if board is None:
            board = self.board
            # 캐시된 결과 확인 (self.board에 대해서만 캐시 사용)
            cache_key = (row, col, num)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Check row
        if num in board[row, :]:
            if board is self.board:
                self._cache[cache_key] = False
            return False
        # Check column
        if num in board[:, col]:
            if board is self.board:
                self._cache[cache_key] = False
            return False
        # Check box
        box_size = int(np.sqrt(board.shape[0]))
        start_row, start_col = box_size * (row // box_size), box_size * (col // box_size)
        if num in board[start_row:start_row + box_size, start_col:start_col + box_size]:
            if board is self.board:
                self._cache[cache_key] = False
            return False
        
        if board is self.board:
            self._cache[cache_key] = True
        return True

    def get_possible_actions(self, state):
        board = self.get_board_from_state(state)
        empty_cells = self.get_empty_cells_from_board(board)
        if not empty_cells:
            return []
        
        actions = []
        for r, c in empty_cells:
            for num in range(1, self.board_size + 1):
                if self.is_valid(r, c, num, board):  # is_valid_on_board 대신 is_valid 사용
                    actions.append(((r, c), num))
        return actions

    def step(self, action):
        row, col = action[0]
        num = action[1]

        # 잘못된 위치에 숫자를 놓으려고 하면 즉시 종료
        if self.board[row, col] != 0:
            return self.board.copy(), -1.0, True  # 이미 채워진 칸에 놓으려고 함
        
        # 숫자를 놓기 전에 유효성 검사
        if not self.is_valid(row, col, num):
            return self.board.copy(), -1.0, True  # 스도쿠 규칙 위반
        
        # 숫자를 놓고 보상 계산
        self.board[row, col] = num
        self._clear_related_cache(row, col)  # 관련된 캐시만 초기화
        
        # 부분 보상 계산
        reward = self._calculate_reward(row, col)
        
        # 퍼즐 완성 여부 확인
        done = len(self.get_empty_cells()) == 0
        if done:
            if SudokuEnv.is_fully_solved(self.board):
                reward += 10.0  # 퍼즐 완성 보상 증가
            else:
                reward = -10.0  # 잘못된 퍼즐 완성 페널티 증가
            
        return self.board.copy(), reward, done

    def _calculate_reward(self, row, col):
        reward = 0.0
        
        # 행 완성 체크
        if 0 not in self.board[row, :]:
            if len(set(self.board[row, :])) == self.board_size:  # 중복 숫자 없는지 확인
                reward += 1.0  # 보상 증가
            else:
                return -1.0  # 중복 숫자가 있으면 페널티
        
        # 열 완성 체크
        if 0 not in self.board[:, col]:
            if len(set(self.board[:, col])) == self.board_size:  # 중복 숫자 없는지 확인
                reward += 1.0  # 보상 증가
            else:
                return -1.0  # 중복 숫자가 있으면 페널티
        
        # 박스 완성 체크
        start_row, start_col = self.box_size * (row // self.box_size), self.box_size * (col // self.box_size)
        box = self.board[start_row:start_row + self.box_size, start_col:start_col + self.box_size]
        if 0 not in box:
            if len(set(box.flatten())) == self.board_size:  # 중복 숫자 없는지 확인
                reward += 1.0  # 보상 증가
            else:
                return -1.0  # 중복 숫자가 있으면 페널티
        
        return reward
    # 정적 메서드는 메서드의 실행이 외부 상태에 영향을 끼치지 않는 순수 함수를 만들 때 사용
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

    def reset(self):
        self.board = self.initial_board.copy()
        self._cache.clear()  # 캐시 초기화
        return self.board.copy()
    
    def get_state(self):
        return tuple(self.board.flatten())

    @staticmethod
    def get_board_from_state(state):
        size = int(np.sqrt(len(state)))
        return np.array(state).reshape((size, size))

    def get_empty_cells_from_board(self, board):
        size = board.shape[0]
        return [(r, c) for r, c in itertools.product(range(size), range(size)) if board[r, c] == 0]

    def _clear_related_cache(self, row, col):
        """특정 위치와 관련된 캐시 항목만 제거"""
        # 해당 행의 캐시 제거
        for c in range(self.board_size):
            for n in range(1, self.board_size + 1):
                self._cache.pop((row, c, n), None)
        
        # 해당 열의 캐시 제거
        for r in range(self.board_size):
            for n in range(1, self.board_size + 1):
                self._cache.pop((r, col, n), None)
        
        # 해당 박스의 캐시 제거
        start_row = self.box_size * (row // self.box_size)
        start_col = self.box_size * (col // self.box_size)
        for r in range(start_row, start_row + self.box_size):
            for c in range(start_col, start_col + self.box_size):
                for n in range(1, self.board_size + 1):
                    self._cache.pop((r, c, n), None) 