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






