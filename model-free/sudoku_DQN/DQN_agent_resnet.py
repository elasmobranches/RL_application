import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models # ResNet 사용을 위해 임포트
   
# --- 신경망 모델 정의 (ResNet-18 사용) ---
class DQNNetwork(nn.Module):
    def __init__(self, grid_size, num_actions):
        super(DQNNetwork, self).__init__()
        self.grid_size = grid_size
        self.num_input_channels = grid_size + 1 # 0(빈칸)부터 grid_size까지의 숫자를 위한 채널

        # ResNet-18 모델 로드 (사전 학습된 가중치 사용 안 함 - weights=None)
        resnet = models.resnet18(weights=None)

        # 1. 첫 번째 컨볼루션 레이어 수정 (self.num_input_channels 입력)
        resnet.conv1 = nn.Conv2d(
            in_channels=self.num_input_channels, # 수정된 입력 채널 수
            out_channels=64, 
            kernel_size=3,   
            stride=1,        
            padding=1,       
            bias=False
        )
        # 2. 초기 MaxPool 레이어 제거 (또는 더 작은 풀링으로 변경)
        resnet.maxpool = nn.Identity() # MaxPool을 Identity로 변경하여 제거

        # 2. 마지막 FC 레이어 교체 (num_actions개의 Q-값 출력)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_actions)

        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1]) # FC 레이어 제외한 모든 레이어
        self.fc_head = resnet.fc # 새로 정의된 FC 레이어

    def forward(self, x_board): # 입력은 (N, H, W) 또는 (H, W) 형태의 보드 (0~grid_size 값)
        # 입력 차원 확인 및 조정 (배치 차원 추가)
        if x_board.dim() == 2: # 단일 상태 (H, W)
            x_board_batched = x_board.unsqueeze(0) 
        elif x_board.dim() == 3: # 배치 상태 (B, H, W)
            x_board_batched = x_board
        else:
            raise ValueError(f"Expected input with 2 or 3 dimensions, got {x_board.dim()}")

        batch_size, H, W = x_board_batched.shape
        
        # 원-핫 인코딩
        # (N, num_channels, H, W) 크기의 텐서 생성
        one_hot_x = torch.zeros(batch_size, self.num_input_channels, H, W, device=x_board.device)
        # scatter_를 사용하여 효율적으로 원-핫 인코딩
        # x_board_batched.unsqueeze(1) -> (N, 1, H, W) for scatter_ index
        # x_board_batched의 값이 0~grid_size 이므로, 인덱스로 바로 사용 가능
        one_hot_x.scatter_(1, x_board_batched.unsqueeze(1).long(), 1)
        
        x = one_hot_x # (N, num_input_channels, H, W)

        # (선택적) 입력 업샘플링: 원-핫 인코딩된 후의 x를 사용
        # if H < 16: 
        #    target_size = max(32, H * 4) 
        #    x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
            
        x = self.resnet_features(x)
        x = torch.flatten(x, 1)
        q_values = self.fc_head(x)
        
        return q_values

# --------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state, dtype=np.float32), \
               np.array(action, dtype=np.int64) - 1, \
               np.array(reward, dtype=np.float32), \
               np.array(next_state, dtype=np.float32), \
               np.array(done, dtype=bool)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, grid_size, learning_rate=0.001, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=10000,
                 replay_buffer_capacity=10000, batch_size=64, target_update_freq=100, use_double_dqn=True):
        
        self.grid_size = grid_size
        self.num_actions = grid_size 
        self.use_double_dqn = use_double_dqn
        
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        if epsilon_decay_steps > 0:
            self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_steps
        else:
            self.epsilon_decay_rate = 0

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_net = DQNNetwork(self.grid_size, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.grid_size, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() 
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.loss_fn = nn.SmoothL1Loss()


    def choose_action(self, state_board, possible_actions_mask):
        self.policy_net.eval() # 행동 선택 시에는 평가 모드로 설정
        with torch.no_grad(): # 평가 모드에서는 그래디언트 계산 불필요
            if random.uniform(0, 1) < self.epsilon:
                valid_indices = np.where(possible_actions_mask)[0] 
                if len(valid_indices) == 0:
                    # self.policy_net.train() # 다시 학습 모드로 돌려놓기 (만약 이 함수가 learn 외부에서 호출된다면)
                    return None 
                chosen_action_index = random.choice(valid_indices) 
                # self.policy_net.train()
                return chosen_action_index + 1 
            else:
                state_tensor = torch.FloatTensor(state_board).to(self.device) 
                q_values = self.policy_net(state_tensor) 
                if q_values.dim() > 1 and q_values.size(0) == 1:
                     q_values = q_values.squeeze(0)

                q_values_masked = q_values.clone()
                q_values_masked[~torch.BoolTensor(possible_actions_mask).to(self.device)] = -float('inf')
                
                chosen_action_index = torch.argmax(q_values_masked).item() 
                # self.policy_net.train() # learn 함수 호출 전에 train() 모드로 복귀해야 함.
                return chosen_action_index + 1
        # 주의: choose_action이 learn 루프 내에서 호출될 경우,
        # learn 시작 시 policy_net.train()을 호출하여 다시 학습 모드로 변경해야 합니다.
        # 또는 choose_action 마지막에 policy_net.train()을 호출합니다. (아래 learn에서 처리)

            
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None # 학습을 수행하지 않았으면 None 반환

        self.policy_net.train() # 학습 시작 전에 모델을 train 모드로 설정
        self.target_net.eval()  # 타겟 네트워크는 항상 eval 모드

        states, actions_0_indexed, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_tensor = torch.FloatTensor(states).to(self.device) 
        actions_tensor = torch.LongTensor(actions_0_indexed).unsqueeze(1).to(self.device) 
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) 
        next_states_tensor = torch.FloatTensor(next_states).to(self.device) 
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)    

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad(): # target_net과 policy_net(Double DQN용)은 그래디언트 계산 불필요
            if self.use_double_dqn:
                # Double DQN: policy_net으로 다음 행동 선택 (eval 모드에서 수행할 수도 있으나, 여기선 train 모드의 policy_net 사용)
                # target_net은 Q값 평가에만 사용
                # 일관성을 위해 policy_net의 다음 행동 예측도 eval 모드에서 하는 것을 고려할 수 있으나,
                # 일반적인 Double DQN 구현은 현재 policy_net을 사용함.
                next_actions_indices = self.policy_net(next_states_tensor).argmax(1, keepdim=True)
                next_max_q_values = self.target_net(next_states_tensor).gather(1, next_actions_indices)
            else:
                next_max_q_values = self.target_net(next_states_tensor).max(1, keepdim=True)[0]
            
            next_max_q_values[dones_tensor] = 0.0

        target_q_values = rewards_tensor + (self.gamma * next_max_q_values)
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss.item() # 계산된 손실 값 반환

    def _update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_rate
            if self.epsilon < self.epsilon_end:
                self.epsilon = self.epsilon_end
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval() 
            self.target_net.eval() 
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")
        except Exception as e:
            print(f"Error loading model: {e}") 