import os
import random
import shutil

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_shared_base(input_shape=(12, 12, 2)):
    """Build the shared convolutional feature extractor for actor and critic."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    return inputs, x


class PPOAgent:
    """Actor-critic PPO agent for the 96-action game board."""

    def __init__(
        self,
        state_shape=(12, 12, 2),
        num_actions=96,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        learning_rate=3e-4,
        entropy_coeff=0.01,
        value_coeff=0.5,
        max_grad_norm=0.5,
        batch_size=64,
        epochs_per_update=5,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update

        inputs, base = build_shared_base(state_shape)
        logits = layers.Dense(num_actions, activation=None)(base)
        value = layers.Dense(1, activation=None)(base)

        self.model = keras.Model(inputs=inputs, outputs=[logits, value])
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer.build(self.model.trainable_variables)
        self.reset_memory()

    def reset_memory(self):
        """Clear the on-policy trajectory buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def remember(self, state, action, reward, done, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def act(self, state, valid_actions, training=True):
        """Choose an action after masking out occupied squares."""
        state_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
        logits, value = self.model(state_tensor, training=False)
        logits = logits.numpy().flatten()
        value = value.numpy().flatten()[0]

        masked_logits = np.full(self.num_actions, -1e9)
        masked_logits[valid_actions] = logits[valid_actions]

        probs = tf.nn.softmax(masked_logits).numpy()
        if training:
            action = np.random.choice(self.num_actions, p=probs)
        else:
            action = valid_actions[np.argmax(probs[valid_actions])]

        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob, value, probs

    def finish_episode(self, last_value=0.0):
        """Compute GAE advantages and update the PPO network."""
        states = np.array(self.states)
        actions = np.array(self.actions, dtype=np.int32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        next_value = last_value
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * mask * last_gae
            next_value = values[t]

        returns = advantages + values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        dataset = tf.data.Dataset.from_tensor_slices(
            (states, actions, advantages, returns, old_log_probs)
        ).shuffle(len(states)).batch(self.batch_size)

        for _ in range(self.epochs_per_update):
            for batch in dataset:
                self._train_step(*batch)

        self.reset_memory()

    def pretrain_policy(self, states, actions, action_masks, epochs=3, batch_size=64):
        """Imitate expert actions with supervised cross-entropy training."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        action_masks = np.asarray(action_masks, dtype=np.float32)

        dataset = tf.data.Dataset.from_tensor_slices(
            (states, actions, action_masks)
        ).shuffle(len(states)).batch(batch_size)

        last_loss = 0.0
        for _ in range(epochs):
            losses = []
            for batch in dataset:
                loss = self._pretrain_policy_step(*batch)
                losses.append(float(loss.numpy()))
            if losses:
                last_loss = float(np.mean(losses))

        return last_loss

    @tf.function
    def _pretrain_policy_step(self, states, actions, action_masks):
        with tf.GradientTape() as tape:
            logits, _ = self.model(states, training=True)
            masked_logits = tf.where(
                action_masks > 0.0,
                logits,
                tf.fill(tf.shape(logits), -1e9),
            )
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=actions,
                    logits=masked_logits,
                )
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [
            tf.zeros_like(var) if grad is None else grad
            for grad, var in zip(grads, self.model.trainable_variables)
        ]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def _train_step(self, states, actions, advantages, returns, old_log_probs):
        with tf.GradientTape() as tape:
            logits, values = self.model(states, training=True)
            values = tf.squeeze(values)

            log_probs_all = tf.nn.log_softmax(logits)
            new_log_probs = tf.reduce_sum(
                log_probs_all * tf.one_hot(actions, self.num_actions), axis=1
            )
            ratio = tf.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = (
                tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            critic_loss = tf.reduce_mean(tf.square(returns - values))

            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * log_probs_all, axis=1)
            entropy_loss = -tf.reduce_mean(entropy)

            loss = (
                actor_loss
                + self.value_coeff * critic_loss
                + self.entropy_coeff * entropy_loss
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath)


class OpponentPool:
    """Stores older policy snapshots for self-play opponents."""

    def __init__(self, max_size=5):
        self.pool = []
        self.max_size = max_size
        self.save_dir = "opponent_pool"
        self.next_id = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def add_model(self, model_weights_path):
        if len(self.pool) >= self.max_size:
            oldest = self.pool.pop(0)
            if os.path.exists(oldest):
                os.remove(oldest)

        dst = os.path.join(self.save_dir, f"opponent_{self.next_id}.weights.h5")
        self.next_id += 1
        shutil.copy(model_weights_path, dst)
        self.pool.append(dst)

    def sample_opponent(self):
        if not self.pool:
            return None
        return random.choice(self.pool)
