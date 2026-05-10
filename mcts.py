import copy
import math
import random


class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = env.get_valid_actions()
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight):
        best_score = -float("inf")
        best_children = []

        for child in self.children:
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.value / child.visits
                explore = exploration_weight * math.sqrt(
                    math.log(max(self.visits, 1)) / child.visits
                )
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)


def _clone_env(env):
    return copy.deepcopy(env)


def _winner_value(winner, root_player):
    if winner == root_player:
        return 1.0
    if winner == -root_player:
        return -1.0
    return 0.0


def _rollout(env, root_player, max_depth):
    depth = 0
    while not env.done_flag and depth < max_depth:
        valid = env.get_valid_actions()
        if not valid:
            break
        env.step(random.choice(valid))
        depth += 1

    if env.done_flag:
        return _winner_value(env.winner, root_player)
    return 0.0


def mcts_action(env, simulations=30, exploration_weight=1.414, rollout_depth=80):
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None
    if len(valid_actions) == 1:
        return valid_actions[0]

    root_player = env.current_player
    root = MCTSNode(_clone_env(env))

    for _ in range(simulations):
        node = root
        sim_env = _clone_env(root.env)

        while node.is_fully_expanded() and node.children and not sim_env.done_flag:
            node = node.best_child(exploration_weight)
            sim_env = _clone_env(node.env)

        if not sim_env.done_flag and node.untried_actions:
            action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
            sim_env.step(action)
            child = MCTSNode(_clone_env(sim_env), parent=node, action=action)
            node.children.append(child)
            node = child

        result = _rollout(sim_env, root_player, rollout_depth)

        while node is not None:
            node.visits += 1
            node.value += result
            node = node.parent

    if not root.children:
        return random.choice(valid_actions)

    return max(root.children, key=lambda child: child.visits).action
