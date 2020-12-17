import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

try:
    import gym
except ImportError:
    raise RuntimeError("Please install opengym: pip install gym")


from ignite.engine import Engine, Events


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(model, observation):
    state = torch.from_numpy(observation).float().unsqueeze(0)
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(model, optimizer, gamma, eps):
    R = 0
    policy_loss = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(model.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_log_probs[:]


EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


def main(env, args):

    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()
    timesteps = list(range(10000))

    def run_single_timestep(engine, timestep):
        observation = engine.state.observation
        action = select_action(model, observation)
        engine.state.observation, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        model.rewards.append(reward)

        if done:
            engine.terminate_epoch()
            engine.state.timestep = timestep

    trainer = Engine(run_single_timestep)

    @trainer.on(Events.STARTED)
    def initialize(engine):
        engine.state.running_reward = 10

    @trainer.on(EPISODE_STARTED)
    def reset_environment_state(engine):
        engine.state.observation = env.reset()

    @trainer.on(EPISODE_COMPLETED)
    def update_model(engine):
        t = engine.state.timestep
        engine.state.running_reward = engine.state.running_reward * 0.99 + t * 0.01
        finish_episode(model, optimizer, args.gamma, eps)

    @trainer.on(EPISODE_COMPLETED(every=args.log_interval))
    def log_episode(engine):
        i_episode = engine.state.epoch
        print(
            f"Episode {i_episode}\tLast length: {engine.state.timestep:5d}\tAverage length: {engine.state.running_reward:.2f}"
        )

    @trainer.on(EPISODE_COMPLETED)
    def should_finish_training(engine):
        running_reward = engine.state.running_reward
        if running_reward > env.spec.reward_threshold:
            print(
                f"Solved! Running reward is now {running_reward} and "
                f"the last episode runs to {engine.state.timestep} time steps!"
            )
            engine.should_terminate = True

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

    env = gym.make("CartPole-v0")
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    main(env, args)
