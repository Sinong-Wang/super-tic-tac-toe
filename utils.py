import numpy as np


def state_to_input(full_board, current_player):
    """
    Convert the 12x12 board into a two-channel neural-network input.

    Channel 0 marks the current player's pieces.
    Channel 1 marks the opponent's pieces.
    """
    board = np.asarray(full_board, dtype=np.int8)
    player = int(current_player)

    my = (board == player).astype(np.float32)
    opp = (board == -player).astype(np.float32)
    return np.stack([my, opp], axis=-1)
