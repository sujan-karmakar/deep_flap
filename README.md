# deep_flap

Flappy Bird reinforcement learning project built on Gymnasium and a small PyTorch DQN.

## Project Files

- [agent.py](agent.py): training and evaluation loop for the Flappy Bird environment
- [dqn.py](dqn.py): the PyTorch network used by the agent
- [experience_replay.py](experience_replay.py): replay buffer implementation
- [flappy_bird_game.py](flappy_bird_game.py): manual Flappy Bird demo controlled with the space bar
- [parameters.yaml](parameters.yaml): named hyperparameter sets used by the agent

## Requirements

- Python 3.10 or newer
- `torch`
- `gymnasium`
- `flappy_bird_gymnasium`
- `pygame`

## Install

Create a virtual environment if you want to keep dependencies isolated, then install the packages:

```bash
pip install torch gymnasium flappy-bird-gymnasium pygame pyyaml
```

## Run

Run the manual game demo:

```bash
python flappy_bird_game.py
```

Run training with one of the hyperparameter names defined in [parameters.yaml](parameters.yaml):

```bash
python agent.py flappybirdv0 --train True
```

Run evaluation/rendering with a saved model:

```bash
python agent.py flappybirdv0
```

## Output

Training writes logs and model checkpoints to [runs/](runs/). Those files are ignored by Git and should not be committed.

## Notes

- The environment name used in the code is `FlappyBird-v0`.
- The training loop saves the best reward seen so far for the selected hyperparameter set.