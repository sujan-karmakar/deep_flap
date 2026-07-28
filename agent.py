import flappy_bird_gymnasium
import gymnasium as gym
import torch
from dqn import DQN
from experience_replay import ReplayMemory

# Device setup
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def run(self, is_training = True, render = False):
    env = gym.make("FlappyBird-v0", render_mode = "human" if render else None)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    policy_dqn = DQN(num_states, num_actions).to(device)

    state, _ = env.reset()

    if is_training:
        memory = ReplayMemory(10000)

    while True:
        action = env.action_space.sample()

        next_state, reward, terminated, _, _ = env.step()

        if is_training:
            memory.append((state, action, next_state, reward, terminated))

        if terminated:
            break

    env.close()