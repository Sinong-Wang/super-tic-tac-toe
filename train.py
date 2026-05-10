import multiprocessing
import random

import numpy as np

from agent import OpponentPool, PPOAgent
from environment import SuperTicTacToeGymEnv
from mcts import mcts_action
from utils import state_to_input


EPISODES = 15000
PARALLEL_WORKERS = 4
UPDATE_EVERY = 4
EVALUATE_EVERY = 500
EVALUATION_GAMES = 100

# AlphaGo-style warm start: imitate MCTS moves before PPO self-play.
USE_MCTS_PRETRAINING = True
PRETRAIN_EXPERT_EXAMPLES = 600
PRETRAIN_MCTS_SIMULATIONS = 20
PRETRAIN_EPOCHS = 3
PRETRAIN_BATCH_SIZE = 64

# Curriculum: first let PPO learn the game, then increase UCT strength smoothly.
MCTS_START_EPISODE = 4000
MCTS_RAMP_1_END = 6999
MCTS_RAMP_2_END = 9999
MCTS_SIM_START = 5
MCTS_SIM_MID = 20
MCTS_SIM_LATE = 50


def mcts_simulations_for_episode(episode):
    if episode < MCTS_START_EPISODE:
        return 0
    if episode <= MCTS_RAMP_1_END:
        progress = (episode - MCTS_START_EPISODE) / (MCTS_RAMP_1_END - MCTS_START_EPISODE)
        return round(MCTS_SIM_START + progress * (MCTS_SIM_MID - MCTS_SIM_START))
    if episode <= MCTS_RAMP_2_END:
        progress = (episode - MCTS_RAMP_1_END) / (MCTS_RAMP_2_END - MCTS_RAMP_1_END)
        return round(MCTS_SIM_MID + progress * (MCTS_SIM_LATE - MCTS_SIM_MID))
    return MCTS_SIM_LATE


def valid_action_mask(valid_actions, action_size=96):
    mask = np.zeros(action_size, dtype=np.float32)
    mask[valid_actions] = 1.0
    return mask


def generate_mcts_expert_data(num_examples, simulations):
    env = SuperTicTacToeGymEnv()
    states = []
    actions = []
    action_masks = []

    while len(states) < num_examples:
        env.reset()
        done = False

        while not done and len(states) < num_examples:
            valid = env.get_valid_actions()
            if not valid:
                break

            expert_action = mcts_action(env, simulations=simulations)
            if expert_action is None:
                break

            states.append(state_to_input(env.full_board, env.current_player))
            actions.append(expert_action)
            action_masks.append(valid_action_mask(valid))

            _, _, done, _, _ = env.step(expert_action)

    return np.array(states), np.array(actions), np.array(action_masks)


def pretrain_with_mcts_expert(agent):
    if not USE_MCTS_PRETRAINING or PRETRAIN_EXPERT_EXAMPLES <= 0:
        return

    print(
        "Generating MCTS expert data: "
        f"{PRETRAIN_EXPERT_EXAMPLES} examples with MCTS({PRETRAIN_MCTS_SIMULATIONS})"
    )
    states, actions, action_masks = generate_mcts_expert_data(
        PRETRAIN_EXPERT_EXAMPLES,
        PRETRAIN_MCTS_SIMULATIONS,
    )
    loss = agent.pretrain_policy(
        states,
        actions,
        action_masks,
        epochs=PRETRAIN_EPOCHS,
        batch_size=PRETRAIN_BATCH_SIZE,
    )
    agent.save("pretrained_ppo.weights.h5")
    print(f"MCTS policy pretraining finished. Final imitation loss: {loss:.4f}")


def worker(current_weights, opponent_weights, mcts_simulations=0):
    env = SuperTicTacToeGymEnv()
    current_agent = PPOAgent()
    current_agent.model.set_weights(current_weights)

    opponent_agent = None
    if mcts_simulations <= 0:
        opponent_agent = PPOAgent()
        if opponent_weights is None:
            opponent_agent.model.set_weights(current_weights)
        else:
            opponent_agent.model.set_weights(opponent_weights)

    state, _ = env.reset()
    env.current_player = 1
    done = False
    trajectory = []

    while not done:
        player = env.current_player
        valid = env.get_valid_actions()
        inp = state_to_input(state, player)

        if player == 1:
            action, logp, val, _ = current_agent.act(inp, valid, training=True)
        elif mcts_simulations > 0:
            action = mcts_action(env, simulations=mcts_simulations)
        else:
            action, _, _, _ = opponent_agent.act(inp, valid, training=False)

        next_state, reward, done, _, _ = env.step(action)

        if player == 1:
            trajectory.append((inp, action, reward, done, val, logp))
        elif done and env.winner == -1 and trajectory:
            s, a, r, d, v, lp = trajectory[-1]
            trajectory[-1] = (s, a, r - 1.0, True, v, lp)

        state = next_state
        if len(trajectory) >= 400:
            break

    return trajectory


def evaluate(agent_weights, num_games=50):
    env = SuperTicTacToeGymEnv()
    agent = PPOAgent()
    agent.model.set_weights(agent_weights)
    wins = 0

    for _ in range(num_games):
        env.reset()
        env.current_player = 1
        done = False

        while not done:
            player = env.current_player
            valid = env.get_valid_actions()
            inp = state_to_input(env.full_board, player)

            if player == 1:
                action, _, _, _ = agent.act(inp, valid, training=False)
            else:
                action = random.choice(valid)

            _, _, done, _, _ = env.step(action)

        if env.winner == 1:
            wins += 1

    return wins / num_games


def train():
    agent = PPOAgent()
    opponent_pool = OpponentPool(max_size=5)

    best_win_rate = -1.0

    pretrain_with_mcts_expert(agent)

    agent.save("current_model.weights.h5")
    opponent_pool.add_model("current_model.weights.h5")

    with multiprocessing.Pool(processes=PARALLEL_WORKERS) as pool:
        trajectories = []

        for ep in range(EPISODES):
            current_weights = agent.model.get_weights()
            mcts_simulations = mcts_simulations_for_episode(ep)

            opp_weights_path = opponent_pool.sample_opponent()
            if opp_weights_path:
                temp_agent = PPOAgent()
                temp_agent.load(opp_weights_path)
                opp_weights = temp_agent.model.get_weights()
            else:
                opp_weights = current_weights

            async_res = pool.apply_async(
                worker,
                (current_weights, opp_weights, mcts_simulations),
            )
            trajectories.append(async_res)

            if len(trajectories) >= UPDATE_EVERY:
                for res in trajectories:
                    traj = res.get()
                    agent.reset_memory()
                    for s, a, r, d, v, lp in traj:
                        agent.remember(s, a, r, d, v, lp)
                    agent.finish_episode(last_value=0.0)
                trajectories = []

            if ep > 0 and ep % EVALUATE_EVERY == 0:
                agent.save("current_model.weights.h5")
                opponent_pool.add_model("current_model.weights.h5")
                win_rate = evaluate(agent.model.get_weights(), num_games=EVALUATION_GAMES)
                opponent_name = (
                    f"MCTS({mcts_simulations})"
                    if mcts_simulations > 0
                    else "opponent-pool"
                )

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    agent.save("best_ppo.weights.h5")
                    best_note = " | new best"
                else:
                    best_note = ""

                print(
                    f"Episode {ep}, opponent={opponent_name}, "
                    f"win rate vs random={win_rate:.2f}, "
                    f"best={best_win_rate:.2f}{best_note}"
                )

    agent.save("final_ppo.weights.h5")
    print(f"Training finished. Best win rate vs random: {best_win_rate:.2f}")


if __name__ == "__main__":
    train()
