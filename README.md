# Reinforcement Learning for Triangular Super Tic-Tac-Toe

This project trains a deep reinforcement learning agent to play a custom **Super Tic-Tac-Toe** game. The game is similar to normal tic-tac-toe, but the board is arranged as a triangle of six `4 x 4` boards. The agent is trained with **PPO**, improved by **self-play**, and strengthened with a lightweight **MCTS/UCT curriculum** inspired by AlphaGo-style training.

---

## Game Description

The full board is represented as a `12 x 12` grid, but only six `4 x 4` regions are playable.

The board layout is:

```text
Level 1:        [board 0]

Level 2:    [board 1] [board 2]

Level 3: [board 3] [board 4] [board 5]

The board offsets inside the 12 x 12 grid are:
board_offsets = {
    0: (0, 4),
    1: (4, 2),
    2: (4, 6),
    3: (8, 0),
    4: (8, 4),
    5: (8, 8),
}
