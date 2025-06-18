# 4x4 스도쿠 DP 기반 강화학습 프로젝트

#### model-free 폴더에는 중간 발표 이전의 내용이 담겨있습니다.
#### sudoku_DP 폴더에는 중간 발표 이후 개선 된 model-based 내용이 담겨 있습니다.

이 폴더는 4x4 스도쿠 퍼즐을 동적계획법(DP, Policy Iteration/Value Iteration) 기반으로 해결하는 방법을 실험합니다.

## 폴더 구성 및 주요 파일 설명

- **agent_DP.py**
  - Policy Iteration, Value Iteration 알고리즘을 구현한 에이전트 클래스가 포함되어 있습니다.
  - `PolicyIterationAgent`: 정책 반복(Policy Iteration) 방식으로 최적 정책을 학습합니다.
  - `ValueIterationAgent`: 가치 반복(Value Iteration) 방식으로 최적 정책을 유도합니다.

- **sudoku_env_DP.py**
  - 4x4 스도쿠 환경을 정의합니다.
  - 상태(state): 4x4 보드의 각 칸 숫자(0은 빈칸)
  - 행동(action): 빈 칸에 1~4 중 하나의 숫자를 넣는 것
  - 보상(reward): 올바른 숫자 입력, 행/열/박스 완성, 퍼즐 완성/실패 등에 따라 차등 지급

- **main_DP.py**
  - 전체 실험 파이프라인을 담당합니다.
  - 환경 및 에이전트 초기화, 학습 루프, 퍼즐 해결, 결과 시각화(히트맵) 등을 포함합니다.
  - 여러 퍼즐에 대해 반복 학습 및 테스트가 가능합니다.
  - 실행 결과는 txt 파일로 저장되고, value heatmap은 이미지로 저장됩니다.

- **utils_DP.py**
  - 퍼즐 데이터(EXAMPLE_PUZZLES_4X4, TEST_PUZZLES_4X4)와 보드 출력 함수(`print_board`) 등 보조 기능 제공

## 실행 방법

```bash
python main_DP.py
```

- 실행 결과는 `sudoku_DP_output_YYYYMMDD_HHMMSS.txt`로 저장됩니다.
- value heatmap 이미지는 `value_heatmap/` 폴더에 저장됩니다.

## 주요 알고리즘 개요

- **Policy Iteration**: 정책 평가와 정책 개선을 반복하여 최적 정책을 찾는 DP 알고리즘
- **Value Iteration**: 가치 함수만 반복적으로 갱신하여 최적 정책을 유도하는 DP 알고리즘

## 참고 사항

- 본 코드는 4x4 스도쿠에 최적화되어 있습니다.
- 6x6 이상의 스도쿠는 상태 공간이 기하급수적으로 커져 현실적으로 DP 적용이 어렵습니다.

# 기타
- Value Comparison은 큰 의미를 갖지는 않습니다