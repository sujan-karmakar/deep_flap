import flappy_bird_gymnasium
import gymnasium as gym
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

        for episode in itertools.count():
            state, _ = env.reset()
            episode_rewards = 0
            terminated = False

            while not terminated:
                action = env.action_space.sample()

                next_state, reward, terminated, _, _ = env.step()

                if is_training:
                    memory.append((state, action, next_state, reward, terminated))

                state = next_state
                episode_reward += reward

            print(f"Episode: {episode + 1}, Reward: {episode_reward}")

        #env.close() # Stop manually