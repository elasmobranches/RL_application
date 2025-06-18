import numpy as np

# 학습에 사용되는 4x4 스도쿠 퍼즐 예제 (0은 빈칸)
# 난이도별로 구성된 퍼즐들
EXAMPLE_PUZZLES_4X4 = [
    {
        "puzzle": np.array([
            [1, 0, 3, 0],
            [0, 0, 2, 1],
            [0, 1, 0, 0],
            [2, 4, 0, 3]
        ]),"difficulty": "Easy"},
        {"puzzle": np.array([
            [0, 0, 0, 0],
            [0, 0, 3, 4],
            [3, 1, 0, 0],
            [0, 0, 0, 4]
        ]), "difficulty": "Medium"},    
        {"puzzle": np.array([
            [0, 1, 0, 4],
            [0, 3, 1, 2],
            [0, 2, 0, 0],
            [0, 0, 0, 1]
        ]), "difficulty": "Medium"},
        {"puzzle": np.array([
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0] 
        ]),"difficulty": "Difficult"}
]

# 테스트에 사용되는 4x4 스도쿠 퍼즐 (0은 빈칸)
# 난이도별로 구성된 테스트 퍼즐들
TEST_PUZZLES_4X4 = [
    {
        "puzzle": np.array([
            [4,2,1,0],
            [1,3,4,2],
            [0,4,3,0],
            [0,0,0,0]
        ]),"difficulty": "Easy"},
    {"puzzle": np.array([
            [1,0,0,3],
            [0,2,0,0],
            [4,1,0,0],
            [0,0,1,0]
        ]),"difficulty": "Medium"},
    {"puzzle": np.array([
            [0,0,0,0],
            [3,0,0,2],
            [0,0,4,0],
            [0,0,0,3]
        ]),"difficulty": "Hard"},
    {"puzzle": np.array([
            [4,3,2,1],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ]),"difficulty": "Difficult"}
]

# 스도쿠 보드를 보기 좋게 출력하는 함수
def print_board(board):
    """
    주어진 스도쿠 보드를 보기 좋게 출력합니다.
    0은 빈 칸을 의미합니다.
    """
    if board is None:
        print("Board is None.")
        return
    
    board_size = board.shape[0]
    box_size = int(np.sqrt(board_size))
    
    for i in range(board_size):
        if i > 0 and i % box_size == 0:
            print("-" * (board_size * 2 + box_size))
        for j in range(board_size):
            if j > 0 and j % box_size == 0:
                print("| ", end="")
            
            cell = board[i, j]
            print(f"{cell if cell != 0 else '.'} ", end="")
        print()
    print()

# 메인 실행 부분 - 예제 퍼즐과 테스트 퍼즐 출력
if __name__ == '__main__':
    print("Available 4x4 Sudoku Puzzles:")
    for i, p_data in enumerate(EXAMPLE_PUZZLES_4X4):
        print(f"--- Puzzle {i+1} ({p_data['difficulty']}) ---")
        print_board(p_data['puzzle'])
        
        
    print("\n--- Test Puzzles ---")
    for i, p_data in enumerate(TEST_PUZZLES_4X4):
        print(f"--- Test Puzzle {i+1} ({p_data['difficulty']}) ---")
        print_board(p_data['puzzle'])
        