import random

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    class _FallbackEnv:
        def reset(self, seed=None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

    class gym:
        Env = _FallbackEnv

    class spaces:
        Discrete = _Discrete
        Box = _Box


class SuperTicTacToeGymEnv(gym.Env):
    """
    Gym-style environment for triangular Super Tic-Tac-Toe.

    The playable area is represented inside a 12x12 grid:
    - Level 1: one 4x4 board at (0, 4)
    - Level 2: two 4x4 boards at (4, 2) and (4, 6)
    - Level 3: three 4x4 boards at (8, 0), (8, 4), and (8, 8)

    Win rules:
    - 4 in a row horizontally
    - 4 in a column, with at least one move in a different level
    - 5 across either diagonal
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.size = 12
        self.num_boards = 6
        self.rows = 4
        self.cols = 4
        self.action_size = self.num_boards * self.rows * self.cols

        self.board_offsets = {
            0: (0, 4),
            1: (4, 2),
            2: (4, 6),
            3: (8, 0),
            4: (8, 4),
            5: (8, 8),
        }

        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.size, self.size), dtype=np.int8
        )

        self.full_board = None
        self.current_player = None
        self.done_flag = False
        self.winner = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.full_board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1 if random.random() < 0.5 else -1
        self.done_flag = False
        self.winner = None
        return self._get_obs(), {}

    def _get_obs(self):
        return self.full_board.copy()

    def get_valid_actions(self):
        """Return action ids for all empty playable squares."""
        actions = []
        for bid, (br, bc) in self.board_offsets.items():
            for r in range(self.rows):
                for c in range(self.cols):
                    if self.full_board[br + r, bc + c] == 0:
                        actions.append(bid * 16 + r * 4 + c)
        return actions

    def step(self, action):
        if self.done_flag:
            return self._get_obs(), 0.0, True, False, {}

        bid = action // 16
        rem = action % 16
        r = rem // 4
        c = rem % 4
        br, bc = self.board_offsets[bid]
        gx, gy = br + r, bc + c

        if self.full_board[gx, gy] != 0:
            self._switch_player()
            return self._get_obs(), -0.1, False, False, {}

        if random.random() < 0.5:
            final_gx, final_gy = gx, gy
            success = True
        else:
            neighbor = self._sample_neighbor(gx, gy)
            if self._is_empty_playable_square(*neighbor):
                final_gx, final_gy = neighbor
                success = True
            else:
                success = False

        shaped_reward = 0.0
        win = False
        draw = False

        if success:
            self.full_board[final_gx, final_gy] = self.current_player

            if self._has_n_in_a_row(self.current_player, 3):
                shaped_reward += 0.2

            if self._check_win(self.current_player):
                win = True
            elif len(self.get_valid_actions()) == 0:
                draw = True
        else:
            shaped_reward = -0.1

        if win:
            reward = 1.0
            self.done_flag = True
            self.winner = self.current_player
        elif draw:
            reward = 0.0
            self.done_flag = True
            self.winner = 0
        else:
            reward = shaped_reward

        if not self.done_flag:
            self._switch_player()

        return self._get_obs(), reward, self.done_flag, False, {}

    def _switch_player(self):
        self.current_player *= -1

    def _get_level(self, x, y):
        """Return the level index for a coordinate, or -1 outside the boards."""
        for bid, (br, bc) in self.board_offsets.items():
            if br <= x < br + 4 and bc <= y < bc + 4:
                if bid == 0:
                    return 0
                if bid in (1, 2):
                    return 1
                return 2
        return -1

    def _sample_neighbor(self, gx, gy):
        """Sample one of the 8 adjacent squares before checking legality."""
        dx, dy = random.choice(
            [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]
        )
        return gx + dx, gy + dy

    def _is_empty_playable_square(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
        return self._get_level(x, y) != -1 and self.full_board[x, y] == 0

    def _check_win(self, player):
        if self._check_direction(player, dx=0, dy=1, target=4, require_cross_level=False):
            return True
        if self._check_direction(player, dx=1, dy=0, target=4, require_cross_level=True):
            return True
        if self._check_direction(player, dx=1, dy=1, target=5, require_cross_level=False):
            return True
        if self._check_direction(player, dx=1, dy=-1, target=5, require_cross_level=False):
            return True
        return False

    def _check_direction(self, player, dx, dy, target, require_cross_level):
        for start_x in range(self.size):
            for start_y in range(self.size):
                if self.full_board[start_x, start_y] != player:
                    continue

                count = 1
                levels = {self._get_level(start_x, start_y)}
                x, y = start_x + dx, start_y + dy

                while (
                    0 <= x < self.size
                    and 0 <= y < self.size
                    and self.full_board[x, y] == player
                ):
                    count += 1
                    levels.add(self._get_level(x, y))
                    x += dx
                    y += dy

                if count >= target:
                    if require_cross_level and len(levels) < 2:
                        continue
                    return True

        return False

    def _has_n_in_a_row(self, player, n):
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for start_x in range(self.size):
                for start_y in range(self.size):
                    if self.full_board[start_x, start_y] != player:
                        continue

                    count = 1
                    x, y = start_x + dx, start_y + dy

                    while (
                        0 <= x < self.size
                        and 0 <= y < self.size
                        and self.full_board[x, y] == player
                    ):
                        count += 1
                        x += dx
                        y += dy

                    if count >= n:
                        return True

        return False

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError

        print("\n" + "=" * 36)
        for x in range(self.size):
            row_str = ""
            for y in range(self.size):
                if self._get_level(x, y) == -1:
                    row_str += "   "
                else:
                    val = self.full_board[x, y]
                    if val == 1:
                        row_str += " O "
                    elif val == -1:
                        row_str += " X "
                    else:
                        row_str += " . "
            print(row_str)
        print("=" * 36)
        print(f"Current player: {'O' if self.current_player == 1 else 'X'}")
        if self.done_flag:
            if self.winner == 0:
                print("Game over: draw")
            else:
                print(f"Game over: winner {self.winner}")

    def close(self):
        pass
