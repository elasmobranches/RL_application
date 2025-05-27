import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchvision import models # ResNet을 사용하지 않으므로 이 라인 주석 처리 또는 제거

# --- 신경망 모델 정의 (가벼운 CNN 사용) ---
class DQNNetwork(nn.Module):
    def __init__(self, grid_size, num_actions): # grid_size=4, num_actions=4 인 경우
        super(DQNNetwork, self).__init__()
        self.grid_size = grid_size
        self.num_input_channels = grid_size + 1 # 0(빈칸)부터 grid_size까지의 숫자를 위한 채널 (예: 5)

        # 입력: (N, num_input_channels, grid_size, grid_size)
        # 예: (N, 5, 4, 4)
        self.conv1 = nn.Conv2d(self.num_input_channels, 32, kernel_size=3, stride=1, padding=1)
        # Conv1 출력: (N, 32, grid_size, grid_size) -> (N, 32, 4, 4)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Conv2 출력: (N, 64, grid_size, grid_size) -> (N, 64, 4, 4)

        # (선택적) 풀링 레이어 - 여기서는 사용하지 않음
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        # 풀링 사용 시 Conv2 출력 후: (N, 64, grid_size/2, grid_size/2) -> (N, 64, 2, 2)

        # FC 레이어 입력 크기 계산
        # 풀링 미사용 시: 64 * grid_size * grid_size
        # 풀링 사용 시 (kernel=2, stride=2): 64 * (grid_size/2) * (grid_size/2)
        if grid_size == 4: # 예시
            fc_input_features = 64 * 4 * 4 # 풀링 미사용 시 1024
            # fc_input_features = 64 * 2 * 2 # 풀링 사용 시 256
        else: # 일반적인 경우
            # 이 부분은 grid_size에 따라 동적으로 계산하는 것이 더 좋음
            # 또는 forward에서 conv 레이어 통과 후 크기를 가져와서 계산
            # 여기서는 일단 grid_size가 4라고 가정하고 값을 직접 설정
            fc_input_features = 64 * grid_size * grid_size # 풀링 미사용 가정

        self.fc1 = nn.Linear(fc_input_features, 256) # FC 레이어 뉴런 수 (예: 256 또는 128)
        self.fc2 = nn.Linear(256, num_actions)

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
        one_hot_x = torch.zeros(batch_size, self.num_input_channels, H, W, device=x_board.device)
        one_hot_x.scatter_(1, x_board_batched.unsqueeze(1).long(), 1)
        x = one_hot_x # (N, num_input_channels, H, W)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = self.pool(x) # 풀링 레이어 사용 시

        x = torch.flatten(x, 1) # 또는 x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
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
        self.policy_net.eval() 
        with torch.no_grad(): 
            if random.uniform(0, 1) < self.epsilon:
                valid_indices = np.where(possible_actions_mask)[0] 
                if len(valid_indices) == 0:
                    return None 
                chosen_action_index = random.choice(valid_indices) 
                return chosen_action_index + 1 
            else:
                state_tensor = torch.FloatTensor(state_board).to(self.device) 
                q_values = self.policy_net(state_tensor) 
                if q_values.dim() > 1 and q_values.size(0) == 1: 
                     q_values = q_values.squeeze(0)

                q_values_masked = q_values.clone()
                q_values_masked[~torch.BoolTensor(possible_actions_mask).to(self.device)] = -float('inf')
                
                chosen_action_index = torch.argmax(q_values_masked).item() 
                return chosen_action_index + 1
        
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None 

        self.policy_net.train() 
        self.target_net.eval()  

        states, actions_0_indexed, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states_tensor = torch.FloatTensor(states).to(self.device) 
        actions_tensor = torch.LongTensor(actions_0_indexed).unsqueeze(1).to(self.device) 
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(self.device) 
        next_states_tensor = torch.FloatTensor(next_states).to(self.device) 
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(self.device)    

        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad(): 
            if self.use_double_dqn:
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
        
        return loss.item() 

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