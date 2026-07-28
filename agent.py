import flappy_bird_gymnasium
import gymnasium as gym
import random
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

# Device setup
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Agent:
    def __init__(self, param_set):
        self.param_set = param_set
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
        
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss = nn.MSELoss()
        self.optimizer = None



    def run(self, is_training = True, render = False):
        env = gym.make("FlappyBird-v0", render_mode = "human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = policy_dqn = DQN(num_states, num_actions).to(device)
            # Copy weight and bias value from Policy to Target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps = 0
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float, device = device)

            episode_reward = 0
            terminated = False

            while not terminated:
                if is_training and random.random() < self.epsilon:
                    action = env.action_space.sample() # Explore
                    action = torch.tensor(action, dtype = torch.long, device = device)
                else:
                    with torch.no_grad:
                        action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax() # Exploit

                next_state, reward, terminated, _, _ = env.step(action.item())

                # Tensors
                reward = torch.tensor(reward, dtype = torch.float, device = device)
                next_state = torch.tensor(next_state, dtype = torch.float, device = device)

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))
                    steps += 1

                state = next_state
                episode_reward += reward

            print(f"Episode: {episode + 1}, Reward: {episode_reward}")

            if is_training:
                # Epsilon Decay:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if len(memory) > self.mini_batch_size:
                    # Get sample experiences
                    mini_batch = memory.sample(self.mini_batch_size)

                    optimize(mini_batch, policy_dqn, target_dqn)

                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

        #env.close() # Stop manually

    
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # Get experience
        state, action, next_state, reward, terminated = zip(*mini_batch)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        terminated = torch.stack(terminated)

        # Target Q
        with torch.no_grad():
            target_q = reward + (1 - terminated) * self.gamma * target_dqn(next_state).max(dim = 1)[0]

        # Predicted Q
        current_q = policy_dqn(state)

        # Loss
        loss = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

