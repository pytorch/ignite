import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from ignite.engine import Engine, Events

try:
    import gymnasium as gym
except ImportError:
    raise ModuleNotFoundError("Please install opengym: pip install gymnasium")


eps = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(policy, observation):
    state = torch.from_numpy(observation).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, gamma):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


def main(env, args):
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    timesteps = range(10000)

    def run_single_timestep(engine, timestep):
        observation = engine.state.observation
        action = select_action(policy, observation)
        engine.state.observation, reward, done, _, _ = env.step(action)
        if args.render:
            env.render()
        policy.rewards.append(reward)
        engine.state.ep_reward += reward
        if done:
            engine.terminate_epoch()
            engine.state.timestep = timestep

    trainer = Engine(run_single_timestep)
    trainer.state.running_reward = 10

    @trainer.on(EPISODE_STARTED)
    def reset_environment_state():
        torch.manual_seed(args.seed + trainer.state.epoch)
        trainer.state.observation, _ = env.reset(seed=args.seed + trainer.state.epoch)
        trainer.state.ep_reward = 0

    @trainer.on(EPISODE_COMPLETED)
    def update_model():
        trainer.state.running_reward = 0.05 * trainer.state.ep_reward + (1 - 0.05) * trainer.state.running_reward
        finish_episode(policy, optimizer, args.gamma)

    @trainer.on(EPISODE_COMPLETED(every=args.log_interval))
    def log_episode():
        i_episode = trainer.state.epoch
        print(
            f"Episode {i_episode}\tLast reward: {trainer.state.ep_reward:.2f}"
            f"\tAverage length: {trainer.state.running_reward:.2f}"
        )

    @trainer.on(EPISODE_COMPLETED)
    def should_finish_training():
        running_reward = trainer.state.running_reward
        if running_reward > env.spec.reward_threshold:
            print(
                f"Solved! Running reward is now {running_reward} and "
                f"the last episode runs to {trainer.state.timestep} time steps!"
            )
            trainer.should_terminate = True

    trainer.run(timesteps, max_epochs=args.max_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch REINFORCE example")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G", help="discount factor (default: 0.99)")
    parser.add_argument("--seed", type=int, default=543, metavar="N", help="random seed (default: 543)")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval", type=int, default=10, metavar="N", help="interval between training status logs (default: 10)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000000,
        metavar="N",
        help="Number of episodes for the training (default: 1000000)",
    )
    args = parser.parse_args()

    env = gym.make("CartPole-v1")

    main(env, args)
