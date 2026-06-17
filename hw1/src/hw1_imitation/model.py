"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.linear1 = nn.Linear(state_dim, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], chunk_size * action_dim)
        self.relu = nn.ReLU()

    def compute_loss(
        self,
        # state: (B, state_dim)
        state: torch.Tensor,
        # action_chunk: (B, chunk_size, action_dim)
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action = self.sample_actions(state)
        return torch.mean((action - action_chunk) ** 2)
    
    # 'view' only works when the requested shape is compatible with the tensor's current memory layout.
    # 'reshape' uses a view when possible, but copies the data if necessary, so it also works with non-contiguous tensors.
    # To use 'view' with a non-contiguous tensor:
    # x.contiguous().view(...)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        out = self.linear3(x)

        return out.reshape(-1, self.chunk_size, self.action_dim)        


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.input_dim = self.state_dim + self.chunk_size * self.action_dim + 1
        self.model = nn.Sequential(
        nn.Linear(self.input_dim, hidden_dims[0]),
        nn.ReLU(),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.ReLU(),
        nn.Linear(hidden_dims[1], self.chunk_size * self.action_dim),
        )

    def compute_loss(
        self,
        # state: (B, state_dim)
        state: torch.Tensor,
        # action_chunk: (B, chunk_dim, action_dim)
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(B, 1, 1, device=action_chunk.device, dtype=action_chunk.dtype,)
        interpolation = tau * action_chunk + (1 - tau) * noise
        
        input = torch.cat([state, interpolation.reshape(B, -1), tau.squeeze(-1)], dim=-1)
        return torch.mean((self.model(input) - (action_chunk - noise).reshape(B, -1)) ** 2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        B = state.shape[0]
        actions = torch.randn(B, self.chunk_size * self.action_dim, device=state.device, dtype=state.dtype,)
        tau = torch.zeros(B, 1, device=state.device, dtype=state.dtype,)
        dt = 1 / num_steps

        for _ in range(num_steps):
            input = torch.cat([state, actions, tau], dim=-1)
            v = self.model(input)
            actions = actions + v * dt
            tau = tau + dt

        return actions.reshape(-1, self.chunk_size, self.action_dim)

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
