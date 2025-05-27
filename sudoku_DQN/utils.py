import numpy as np
import random
import matplotlib.pyplot as plt

# 4x4 스도쿠 퍼즐 예제들 (리스트 형태)
# 각 퍼즐은 2차원 리스트 또는 numpy 배열 형태
# 0은 빈칸을 의미

EXAMPLE_PUZZLES_4X4 = [
    [
        [1, 0, 3, 0],
        [0, 0, 0, 2],
        [0, 1, 0, 0],
        [0, 0, 0, 4]
    ],
    [
        [0, 0, 0, 1],
        [1, 0, 2, 0],
        [0, 3, 0, 4],
        [4, 0, 0, 0]
    ],
    [
        [2, 0, 0, 0],
        [0, 1, 0, 3],
        [0, 4, 0, 0],
        [3, 0, 2, 0]
    ],
    [
        [0, 2, 4, 0],
        [0, 0, 0, 0],
        [3, 0, 0, 2],
        [0, 1, 0, 0]
    ],
    # 필요에 따라 더 많은 퍼즐 추가 가능
]

def get_random_puzzle_4x4():
    """4x4 예제 퍼즐 중 하나를 무작위로 반환합니다."""
    if not EXAMPLE_PUZZLES_4X4:
        return None
    return np.array(random.choice(EXAMPLE_PUZZLES_4X4)) # numpy 배열로 복사하여 반환

# 시각화를 위한 함수 (matplotlib 사용)
def plot_rewards(episode_rewards, title='Episode Rewards over Time', save_path=None):
    """에피소드별 보상 변화를 시각화하고 파일로 저장합니다."""
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Reward plot saved to {save_path}")
    plt.show()
    plt.close() # Figure 객체 닫기

def plot_losses(episode_avg_losses, title='Average Loss per Episode', save_path=None):
    """에피소드별 평균 손실 변화를 시각화하고 파일로 저장합니다."""
    if not episode_avg_losses:
        print("No loss data to plot.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(episode_avg_losses)
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    plt.show()
    plt.close() # Figure 객체 닫기

if __name__ == '__main__':
    print("4x4 스도쿠 퍼즐 예제:")
    for i, p in enumerate(EXAMPLE_PUZZLES_4X4):
        print(f"Puzzle {i+1}:")
        for row in p:
            print(row)
        print()
    
    print("무작위 퍼즐 선택:")
    random_puzzle = get_random_puzzle_4x4()
    if random_puzzle is not None:
        for row in random_puzzle:
            print(row) 