import argparse
import os
import random
import sys

from environment import SuperTicTacToeGymEnv
from mcts import mcts_action
from utils import state_to_input


def load_agent(model_path):
    if not os.path.exists(model_path):
        print(f"Error: model file '{model_path}' was not found.")
        print("Run train.py first, or check the model path.")
        sys.exit(1)

    from agent import PPOAgent

    agent = PPOAgent()
    agent.load(model_path)
    return agent


def choose_action(env, player_type, agent=None, simulations=200):
    valid = env.get_valid_actions()

    if player_type == "random":
        return random.choice(valid)

    if player_type == "mcts":
        return mcts_action(env, simulations=simulations)

    if player_type == "ppo":
        inp = state_to_input(env.full_board, env.current_player)
        action, _, _, _ = agent.act(inp, valid, training=False)
        return action

    raise ValueError(f"Unknown player type: {player_type}")


def play_against_human(ai_type="ppo", model_path="final_ppo.weights.h5", simulations=200):
    agent = load_agent(model_path) if ai_type == "ppo" else None
    env = SuperTicTacToeGymEnv()

    env.reset()
    env.current_player = 1
    done = False

    while not done:
        env.render()
        valid = env.get_valid_actions()

        if env.current_player == 1:
            action = choose_action(env, ai_type, agent, simulations)
            print(f"AI action: {action}")
        else:
            while True:
                try:
                    action = int(input("Your action (0-95): "))
                except ValueError:
                    print("Please enter a number.")
                    continue

                if action in valid:
                    break
                print("Invalid action, please choose an empty playable square.")

        _, _, done, _, _ = env.step(action)

    env.render()
    if env.winner == 1:
        print("AI wins.")
    elif env.winner == -1:
        print("You win.")
    else:
        print("Draw.")


def evaluate_agent(
    ai_type="ppo",
    opponent_type="random",
    model_path="final_ppo.weights.h5",
    games=100,
    simulations=200,
):
    agent = load_agent(model_path) if ai_type == "ppo" else None
    opponent_agent = load_agent(model_path) if opponent_type == "ppo" else None
    env = SuperTicTacToeGymEnv()
    wins = 0
    losses = 0
    draws = 0

    for _ in range(games):
        env.reset()
        env.current_player = 1
        done = False

        while not done:
            if env.current_player == 1:
                action = choose_action(env, ai_type, agent, simulations)
            else:
                action = choose_action(env, opponent_type, opponent_agent, simulations)

            _, _, done, _, _ = env.step(action)

        if env.winner == 1:
            wins += 1
        elif env.winner == -1:
            losses += 1
        else:
            draws += 1

    print(f"AI: {ai_type}, opponent: {opponent_type}")
    print(f"Games: {games}, wins: {wins}, losses: {losses}, draws: {draws}")
    print(f"Win rate: {wins / games:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["evaluate", "play"], default="evaluate")
    parser.add_argument("--ai", choices=["ppo", "mcts", "random"], default="ppo")
    parser.add_argument("--opponent", choices=["random", "mcts", "ppo"], default="random")
    parser.add_argument("--model", default="final_ppo.weights.h5")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--simulations", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "play":
        play_against_human(args.ai, args.model, args.simulations)
    else:
        evaluate_agent(args.ai, args.opponent, args.model, args.games, args.simulations)
