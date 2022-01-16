"""
Edited from https://towardsdatascience.com/deep-q-network-with-pytorch-146bfa939dfe
Thank you for amazing post!
"""

from functools import partial
from typing import Any, Dict, List, Tuple, Type
import random

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gdata
from torch_geometric.nn import global_max_pool

from dqn.model import GraphEdgeConvEmb
from utils import state_to_pyg_data, g_argmax, g_gather_index

# TODO: Too messy... need more refactoring.
class GreedyVertexSelector:
    def __init__(
        self,
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
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # DQN network
        self.dqn = GraphEdgeConvEmb(
            hidden_channels=64,
            input_vert_channels=2,
            input_vert_n_vocab=4,
            grow_size=1.3,
            n_layers=6,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size

        self.STATE_PRV_MEM = [0 for _ in range(self.max_memory_size)]
        self.STATE_AFT_MEM = [0 for _ in range(self.max_memory_size)]

        self.ACTION_MEM = torch.zeros(max_memory_size, 1).long()
        self.REWARD_MEM = torch.zeros(max_memory_size, 1)

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

    def _q_function(self, pyg_data):
        q = self.dqn(
            x=pyg_data.x,
            edge_index=pyg_data.edge_index,
            x_emb=pyg_data.x_occ,
            batch=pyg_data.batch,
        )

        return q

    def remember(self, state, action, reward, state2, done):
        """Store the experiences in a buffer to use later"""
        self.STATE_PRV_MEM[self.ending_position] = state_to_pyg_data(state)
        self.STATE_AFT_MEM[self.ending_position] = state_to_pyg_data(state2)

        self.ACTION_MEM[self.ending_position] = action
        self.REWARD_MEM[self.ending_position] = reward.float()

        self.DONE_MEM[self.ending_position] = done.float()
        self.ending_position = (
            self.ending_position + 1
        ) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)

        STATE = gdata.Batch.from_data_list([self.STATE_PRV_MEM[i] for i in idx])

        STATE2 = gdata.Batch.from_data_list([self.STATE_AFT_MEM[i] for i in idx])

        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]

        DONE = self.DONE_MEM[idx]
        return STATE, ACTION, REWARD, STATE2, DONE

    @torch.no_grad()
    def act(self, state: Dict[str, Any]) -> int:

        """Epsilon-greedy action"""

        if random.random() < self.exploration_rate:
            # select index position from where vertex_occupancy is zero
            veroc = state["vertex_occupancy"]
            indexset = np.arange(veroc.shape[0])
            # print(indexset)
            # print(veroc.shape)
            indexset = indexset[veroc == 0]
            ra = random.choice(indexset.tolist())
            # print(ra)
            return ra
        else:

            gdata = state_to_pyg_data(state).to(self.device)
            ac = g_argmax(self._q_function(gdata)).detach().cpu().item()
            # print(ac)
            return ac

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

        maxQ = global_max_pool(self._q_function(STATE2), STATE2.batch)

        # print(maxQ.max(), maxQ.min())

        # print(REWARD.shape, maxQ.shape)
        # print(DONE)
        target = REWARD + self.gamma * maxQ * (1 - DONE)
        current = g_gather_index(self._q_function(STATE), STATE.batch, ACTION)

        # print(current.mean())

        loss = self.l1(current, target)
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Backpropagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
