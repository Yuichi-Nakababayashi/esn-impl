import math
import random
from collections import deque, namedtuple
from itertools import count
from typing import no_type_check_decorator

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from torch.nn.modules import loss

env = gym.make("CartPole-v0").unwrapped

# set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, in_dim, h_dims, out_dim):
        super().__init__()
        dims = [in_dim] + h_dims + [out_dim]
        layers = [
            [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()] for i in range(len(dims) - 1)
        ]
        layers = sum(layers, [])
        layers.append(nn.Flatten())
        self.q_net = nn.Sequential(*layers)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        return self.q_net(x)

    def save_net(self, dir: str):
        """save model weight to the given directory"""
        assert dir[-4:] == ".pth", "INVALID FILE FMT"
        torch.save(self.q_net.state_dict(), dir)

    def load_net(self, dir: str):
        """ "load the save model weight from the given directory"""
        self.q_net.load_state_dict(torch.load(dir))


BATCH_SIZE = 256
SAVE_DIR = "policy_net.pth"
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.00
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()

# Get number of actions from gym action space
n_actions = env.action_space.n
o_space = env.observation_space.shape[0]
h_dims = [16, 16]

policy_net = DQN(o_space, h_dims, n_actions).to(device)
target_net = DQN(o_space, h_dims, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.03)
memory = ReplayMemory(30000)


steps_done = 0


def select_action(state, net_work, is_eval=False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    if is_eval:
        eps_threshold = -1.0
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return net_work(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    #     means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
    plt.savefig("temp.png")
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # take max over the whole action space
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.to("cpu").item()


def evaluate_policy():
    return 0


if __name__ == "__main__":
    num_episodes = 300
    losses = []
    best_duration = 0
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = env.reset()
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        tot_loss = 0.0
        num_loss = 0
        for t in count():
            # Select and perform an action
            action = select_action(state, policy_net)
            next_state, reward, done, _ = env.step(action.item())
            # 倒れるか、200以上続いたら、done
            reward = torch.tensor([reward], device=device)

            # Observe new state
            # at the end of each episode, next_state is None
            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            temp = optimize_model()
            if temp is not None:
                num_loss += 1
                tot_loss += temp
            if done or t + 1 == 100:
                if t + 1 > best_duration:
                    best_duration = t + 1
                    policy_net.save_net(SAVE_DIR)

                episode_durations.append(t + 1)
                plot_durations()
                if num_loss:
                    losses.append(tot_loss / num_loss)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    plt.clf()
    plt.plot(list(range(len(losses))), losses)
    plt.savefig("temp_loss.png")
    print("Complete")
    env.close()
    plt.ioff()
    plt.show()
