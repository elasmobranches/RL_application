# 4x4 스도쿠 DP 기반 강화학습 프로젝트

이 폴더는 4x4 스도쿠 퍼즐을 동적계획법(DP, Policy Iteration/Value Iteration) 기반으로 해결하는 Python 코드 예제입니다.

## 폴더 구성 및 주요 파일 설명

- **agent_DP.py**
  - Policy Iteration, Value Iteration 알고리즘을 구현한 에이전트 클래스가 포함되어 있습니다.
  - `PolicyIterationAgent`: 정책 반복(Policy Iteration) 방식으로 최적 정책을 학습합니다.
  - `ValueIterationAgent`: 가치 반복(Value Iteration) 방식으로 최적 정책을 학습합니다.
  - 각 에이전트는 환경(SudokuEnv)과 상호작용하며, 상태-가치 함수, 정책 갱신, 정책 평가 등의 메서드를 포함합니다.

- **sudoku_env_DP.py**
  - 4x4 스도쿠 환경을 정의합니다.
  - 상태(state)는 4x4 보드의 각 칸의 숫자(0은 빈칸)로 표현됩니다.
  - 행동(action)은 빈 칸에 1~4 중 하나의 숫자를 넣는 것.
  - 보상(reward)은 올바른 숫자 입력, 행/열/박스 완성, 퍼즐 완성/실패 등에 따라 차등 지급됩니다.
  - 환경의 주요 메서드: `step()`, `reset()`, `get_possible_actions()`, `is_valid()`, `is_fully_solved()` 등.

- **main_DP.py**
  - 전체 실험 파이프라인을 담당합니다.
  - 환경 및 에이전트 초기화, 학습 루프, 퍼즐 해결, 결과 시각화(히트맵) 등을 포함합니다.
  - 여러 퍼즐에 대해 점진적 학습 및 테스트를 수행하며, 학습/해결 시간, 성공률 등 성능을 요약 출력합니다.
  - 실행 시 결과가 txt 파일로 저장되고, value heatmap이 이미지로 저장됩니다.

- **utils_DP.py**
  - 퍼즐 데이터(EXAMPLE_PUZZLES_4X4, TEST_PUZZLES_4X4)와 보드 출력 함수(`print_board`) 등 보조 기능을 제공합니다.
  - 퍼즐은 numpy array로 저장되어 있으며, 난이도 정보도 포함되어 있습니다.

## 실행 방법
- main_DP.py 실행:

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
- Policy Iteration은 적절하지 않았습니다.
- 6x6 이상의 스도쿠부터 상태 공간이 기하급수적으로 늘어나 문제 해결에 어려움이 있습니다.