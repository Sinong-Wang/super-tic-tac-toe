# Reinforcement Learning for Triangular Super Tic-Tac-Toe

This project trains a deep reinforcement learning agent to play a custom **Super Tic-Tac-Toe** game. The game is similar to normal tic-tac-toe, but the board is arranged as a triangle of six `4 x 4` boards. The agent is trained with **PPO**, improved by **self-play**, and strengthened with a lightweight **MCTS/UCT curriculum** inspired by AlphaGo-style training.
<img width="1510" height="598" alt="a2816041-95b4-4e49-ae1f-8f150b1a0738" src="https://github.com/user-attachments/assets/3e31d2af-e2ea-4f3b-957b-8d414be124d6" />


---

## Game Description

The full board is represented as a `12 x 12` grid, but only six `4 x 4` regions are playable.

The board layout is:

```text
Level 1:        [board 0]

Level 2:    [board 1] [board 2]

Level 3: [board 3] [board 4] [board 5]
```

The board offsets inside the 12 x 12 grid are:

```text
board_offsets = {
    0: (0, 4),
    1: (4, 2),
    2: (4, 6),
    3: (8, 0),
    4: (8, 4),
    5: (8, 8),
}
```

Each board has size 4 x 4, so there are:
6 boards x 16 cells = 96 possible actions

---

## Game Rules
Players take turns choosing an empty square.

After a player chooses a square:

With probability 1/2, the piece is placed on the chosen square.
With probability 1/2, one of the 8 adjacent squares is sampled.
If the sampled adjacent square is outside the playable board or already occupied, the move is forfeited.
The win conditions are:

4 in a row horizontally.
4 in a column, but at least one piece must be in a different level.
5 in a diagonal direction.

---

## Design Overview
The project uses three main ideas:

1. PPO neural-network agent
2. Self-play with an opponent pool
3. MCTS/UCT as an expert and curriculum opponent
The final training pipeline is:

```text
MCTS expert data
        ↓
Supervised policy pretraining
        ↓
PPO self-play training
        ↓
Light MCTS/UCT curriculum opponent
        ↓
Best model checkpoint
```

This is inspired by the AlphaGo idea:

```text
expert policy learning -> reinforcement learning self-play -> stronger policy
```

Since this custom game has no human expert data, the project uses MCTS-generated actions as artificial expert data.

---

## Neural Network Input
The board is converted into a 12 x 12 x 2 tensor.
```text
Channel 0: current player's pieces
Channel 1: opponent's pieces
```
This representation is created in utils.py.

The playable shape is handled by the environment and action mask, so the agent cannot choose outside-board cells.

---

## PPO Agent
The PPO agent is implemented in agent.py.

The network has:
Two convolutional layers.
One dense hidden layer.
A policy head for action logits.
A value head for state-value prediction.

The agent uses:
PPO clipped objective.
Generalized Advantage Estimation.
Entropy regularization.
Gradient clipping.
Valid-action masking during action selection.

---

## MCTS / UCT
The MCTS logic is implemented in mcts.py.

UCT selection uses:
```text
Q / N + c * sqrt(log(parent_N) / N)
```
where:

Q / N is exploitation.
sqrt(log(parent_N) / N) is exploration.
c = 1.414 by default.
MCTS is used in two ways:

1. Pretraining expert

MCTS generates state-action examples.
The PPO policy learns to imitate these actions before reinforcement learning starts.

2. Training opponent

During PPO training, the opponent gradually becomes stronger by using more MCTS simulations.

---

## MCTS Curriculum
The MCTS opponent strength increases smoothly during training.

Current schedule:
```text
Episode 0-3999:      no MCTS
Episode 4000-6999:   MCTS simulations increase from 5 to 20
Episode 7000-9999:   MCTS simulations increase from 20 to 50
Episode 10000-15000: MCTS simulations stay at 50
```
This avoids a sudden difficulty jump and makes training more stable.

---

## MCTS Curriculum
File Structure
```text
agent.py
```
Defines the PPO neural-network agent, PPO update logic, supervised MCTS pretraining, and opponent pool.

```text
environment.py
```
Defines the game environment, board layout, random placement rule, valid actions, rewards, and win checking.

```text
mcts.py
```
Implements MCTS/UCT action selection.

```text
utils.py
```
Converts the board state into neural-network input.

```text
train.py
```
Runs the full training process:
1. MCTS expert pretraining.
2. PPO training.
3. Opponent-pool self-play.
4. MCTS curriculum opponent.
5. Best-model saving.

```text
evaluate.py
```
Evaluates a trained model or plays against the AI.

---

## Requirements
Install the required packages:
```text
pip install tensorflow numpy gymnasium
```

---

## How to Train
Run:
```text
python train.py
```
Training will create several weight files:
```text
pretrained_ppo.weights.h5
current_model.weights.h5
Opponent File
final_ppo.weights.h5
```

---

## How to Evaluate
Run:
```text
python evaluate.py
```

---

## Training Details
Important training settings in train.py:
```text
EPISODES = 15000   #could be 10000
EVALUATE_EVERY = 500
EVALUATION_GAMES = 100
```
MCTS pretraining settings:
```text
USE_MCTS_PRETRAINING = True
PRETRAIN_EXPERT_EXAMPLES = 600
PRETRAIN_MCTS_SIMULATIONS = 20
PRETRAIN_EPOCHS = 3
```
MCTS curriculum settings:
```text
MCTS_START_EPISODE = 4000
MCTS_SIM_START = 5
MCTS_SIM_MID = 20
MCTS_SIM_LATE = 50
```

---

## Design Motivation
The main challenge is that this game has randomness and no human expert data.

To solve this, the project combines several methods:

1. PPO learns from reward through self-play.
2. Opponent pool prevents the agent from overfitting to only its current version.
3. MCTS pretraining gives the neural network a better starting policy.
4. MCTS curriculum gradually increases opponent difficulty.
5. Best checkpoint saving avoids losing a strong model due to later noisy updates.
This design keeps the final result as a trained deep neural network while still using search to improve learning.

---

## Summary
This project trains a PPO-based neural-network agent for a custom triangular Super Tic-Tac-Toe game. The agent is improved with MCTS expert pretraining, self-play, an opponent pool, and a smooth MCTS curriculum. The final model can be evaluated against random, PPO, or MCTS opponents.











