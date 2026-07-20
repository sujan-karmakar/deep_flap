# deep_flap

This is a minor reinforcement learning project built around a Flappy Bird Gymnasium environment and a simple DQN model.

## What’s in the repo

- `flappy_bird_game.py`: a manual Flappy Bird demo controlled with the space bar
- `dqn.py`: a small PyTorch DQN network definition
- `agent.py`: an environment loop scaffold for running the Flappy Bird Gymnasium environment

## Requirements

- Python 3.10 or newer
- `torch`
- `gymnasium`
- `flappy_bird_gymnasium`
- `pygame`

## Install

Create a virtual environment if you want to keep dependencies isolated, then install the packages:

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame
```

## Run the game

Launch the manual game demo with:

```bash
python flappy_bird_game.py
```

Use the space bar to flap.

## Notes

- The project is set up for DQN experimentation, but the training loop is still a work in progress.
- The environment name used in the code is `FlappyBird-v0`.