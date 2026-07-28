# deep_flap

`deep_flap` is a small reinforcement learning project built around the Flappy Bird Gymnasium environment and a simple Deep Q-Network (DQN) implemented in PyTorch. The repository contains both an automated training agent and a manual game demo for interactive play.

## Overview

The project demonstrates a basic DQN workflow for a Flappy Bird control task. During training, the agent observes environment states, selects actions using an epsilon-greedy policy, stores transitions in replay memory, and updates the network from sampled mini-batches. The best-performing model checkpoint is written to disk so progress can be retained across runs.

## Repository Structure

- [agent.py](agent.py): training and evaluation entry point for the DQN agent
- [dqn.py](dqn.py): neural network definition used by the agent
- [experience_replay.py](experience_replay.py): simple FIFO replay memory implementation
- [flappy_bird_game.py](flappy_bird_game.py): manual Flappy Bird demo controlled with the keyboard
- [parameters.yaml](parameters.yaml): named hyperparameter configurations for training
- [runs/](runs/): generated training logs and model checkpoints

## Requirements

- Python 3.10 or newer
- `torch`
- `gymnasium`
- `flappy_bird_gymnasium`
- `pygame`
- `pyyaml`

## Installation

It is recommended to create and activate a virtual environment before installing dependencies.

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

## Usage

### Manual Play

Run the interactive Flappy Bird demo:

```bash
python flappy_bird_game.py
```

Press the space bar to flap. Close the window to end the session.

### Training

Start training by passing the name of a hyperparameter set defined in [parameters.yaml](parameters.yaml):

```bash
python agent.py flappybirdv0 --train True
```

The `flappybirdv0` argument refers to the configuration key in [parameters.yaml](parameters.yaml). If additional configurations are added, their key should be passed in the same position.

### Evaluation

Run the agent without the training flag to load the saved model and render the environment:

```bash
python agent.py flappybirdv0
```

## Agent Behavior

The training pipeline in [agent.py](agent.py) uses the following components and steps:

- The current environment is created with the Flappy Bird Gymnasium wrapper.
- The state and action dimensions are read directly from the environment.
- A policy network and target network are instantiated from [dqn.py](dqn.py).
- Replay memory is used to store `(state, action, next_state, reward, terminated)` tuples.
- Epsilon-greedy exploration is used during training.
- A mini-batch is sampled once the replay memory contains enough transitions.
- The loss is computed with mean squared error between predicted and target Q-values.
- The best checkpoint is saved whenever the episode reward exceeds the previous best result.

## Model Architecture

[dqn.py](dqn.py) defines a compact feed-forward neural network:

- Input layer sized to the environment state space
- One hidden fully connected layer with ReLU activation
- Output layer sized to the action space

This architecture is intentionally lightweight and suitable for a small control task such as Flappy Bird.

## Hyperparameters

The active configuration is loaded from [parameters.yaml](parameters.yaml). The current file defines the `flappybirdv0` profile with the following values:

- `env_id`: `FlappyBird-v0`
- `epsilon_init`: initial exploration rate
- `epsilon_min`: lower bound for exploration
- `epsilon_decay`: multiplicative epsilon decay factor
- `replay_memory_size`: maximum replay buffer size
- `mini_batch_size`: number of experiences sampled per optimization step
- `network_sync_rate`: number of environment steps between target network synchronizations
- `alpha`: Adam optimizer learning rate
- `gamma`: discount factor
- `reward_threshold`: early stopping threshold for episode reward

## Output Files

Training writes artifacts to [runs/](runs/):

- `*.log`: text log file containing best reward updates
- `*.pt`: PyTorch checkpoint containing the best model parameters seen so far

These files are generated at runtime.

## Runtime Notes

- The environment name used by the code is `FlappyBird-v0`.
- Training can be interrupted with Ctrl+C or if reward threshold is reached.
- When the process is interrupted, the environment is closed cleanly.
- The saved `.pt` file represents the best model found during training.