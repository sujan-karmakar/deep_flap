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
import argparse
import os


# Device setup
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exists_ok = True)


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

        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")



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

            best_reward = float("-inf")

        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float, device = device)

            episode_reward = 0
            terminated = False

            while (episode_reward < self.reward_threshold and not terminated):
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
                # Save best model
                if episode_reward > best_reward:
                    log_msg = f"Best reward: {episode_reward}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Epsilon Decay:
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if len(memory) > self.mini_batch_size:
                    # Get sample experiences
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps = 0

        #env.close() # Stop manually


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Train or test model")
    parser.add_argument("hyperparameters", help = "")
    parser.add_argument("--train")
    args = parser.parse_args()

    dqn = Agent(param_set = args.hyperparameters)

    if args.train:
        dqn.run(is_training = True)
    else:
        dqn.run(is_training = False)