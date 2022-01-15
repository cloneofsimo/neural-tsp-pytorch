"""
Edited from https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe
Thank you for amazing post!
"""

from typing import List, Tuple, Type
import random

import torch
import torch.nn as nn
import torch_geometric.data


from dqn.model import GraphConvEmb

# TODO: Behavior of this buffer is not yet defined.
class GraphBuffer:
    def __init__(self, capacity: int):

        self.capacity = capacity

        self.buffer = []

    def __getitem__(self, key):
        return self.buffer[key]

    def __len__(self):
        return self.capacity

    def __setitem__(self, key, value):
        self.buffer[key] = value


# TODO: Too messy... need more refactoring.
class GreedyVertexSelector:
    def __init__(
        self,
        state_space: List[int],
        action_space: int,
        max_memory_size: int,
        batch_size: int,
        gamma: float = 0.9,
        lr: float = 0.004,
        exploration_max: float = 1.0,
        exploration_min: float = 0.02,
        exploration_decay: float = 0.99,
        pretrained: bool = False,
    ):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # DQN network
        self.dqn = GraphConvEmb().to(self.device)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size

        self.STATE_MEM = GraphBuffer(max_memory_size)
        self.ACTION_MEM = torch.zeros(max_memory_size, 1)
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)
        self.STATE2_MEM = torch.zeros(max_memory_size, self.state_space)
        self.DONE_MEM = torch.zeros(max_memory_size, 1)

        self.ending_position: int = 0
        self.num_in_queue: int = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min

        self.exploration_decay = exploration_decay

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (
            self.ending_position + 1
        ) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    def act(self, state):
        """Epsilon-greedy action"""

        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return (
                torch.argmax(self.dqn(state.to(self.device)))
                .unsqueeze(0)
                .unsqueeze(0)
                .cpu()
            )

    def experience_replay(self):
        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)
        DONE = DONE.to(self.device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + torch.mul(
            (self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE
        )
        current = self.dqn(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
