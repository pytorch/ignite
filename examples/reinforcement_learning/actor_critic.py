import argparse
from collections import deque, namedtuple

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


SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

eps = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


def select_action(policy, observation):
    observation = torch.from_numpy(observation).float()
    probs, observation_value = policy(observation)
    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    policy.saved_actions.append(SavedAction(m.log_prob(action), observation_value))

    # the action to take (left or right)
    return action.item()


def finish_episode(policy, optimizer, gamma):
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = policy.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = deque()  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in policy.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.appendleft(R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()
    # reset rewards and action buffer
    del policy.rewards[:]
    del policy.saved_actions[:]


EPISODE_STARTED = Events.EPOCH_STARTED
EPISODE_COMPLETED = Events.EPOCH_COMPLETED


def main(env, args):
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-2)
    timesteps = range(10000)

    def run_single_timestep(engine, timestep):
        observation = engine.state.observation
        # select action from policy
        action = select_action(policy, observation)

        # take the action
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
        # reset environment and episode reward
        torch.manual_seed(args.seed + trainer.state.epoch)
        trainer.state.observation, _ = env.reset(seed=args.seed + trainer.state.epoch)
        trainer.state.ep_reward = 0

    @trainer.on(EPISODE_COMPLETED)
    def update_model():
        # update cumulative reward
        t = trainer.state.timestep
        trainer.state.running_reward = 0.05 * trainer.state.ep_reward + (1 - 0.05) * trainer.state.running_reward
        # perform backprop
        finish_episode(policy, optimizer, args.gamma)

    @trainer.on(EPISODE_COMPLETED(every=args.log_interval))
    def log_episode():
        i_episode = trainer.state.epoch
        print(
            f"Episode {i_episode}\tLast reward: {trainer.state.ep_reward:.2f}"
            f"\tAverage reward: {trainer.state.running_reward:.2f}"
        )

    @trainer.on(EPISODE_COMPLETED)
    def should_finish_training():
        # check if we have "solved" the cart pole problem
        running_reward = trainer.state.running_reward
        if running_reward > env.spec.reward_threshold:
            print(
                f"Solved! Running reward is now {running_reward} and "
                f"the last episode runs to {trainer.state.timestep} time steps!"
            )
            trainer.should_terminate = True

    trainer.run(timesteps, max_epochs=args.max_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ignite actor-critic example")
    parser.add_argument("--gamma", type=float, default=0.99, metavar="G", help="discount factor (default: 0.99)")
    parser.add_argument("--seed", type=int, default=543, metavar="N", help="random seed (default: 1)")
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
