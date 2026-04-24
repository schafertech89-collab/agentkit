"""Memory policy controller.

A small PPO-style policy that, given the current *memory observation* for a
MemCube (importance, activation_strength, tier-one-hot, utility_score,
regret, days_since_created) emits a discrete action:

    KEEP, PROMOTE, CONSOLIDATE, FORGET, REINFORCE

When torch is available we instantiate the neural network and update with a
clipped PPO objective. When torch isn't available we fall back to a
deterministic rule-based policy so the agent stack still functions.

This is the RL-governed memory policy called out in Architecture 5.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

try:
    import torch
    from torch import nn
    from torch.nn import functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class PolicyAction(IntEnum):
    KEEP = 0
    PROMOTE = 1
    CONSOLIDATE = 2
    FORGET = 3
    REINFORCE = 4


@dataclass
class PolicyDecision:
    action: PolicyAction
    target_tier: str
    decay_rate: float
    log_prob: float
    value: float


if _HAS_TORCH:

    class _Net(nn.Module):
        def __init__(self, obs_dim: int = 9, action_dim: int = 5, hidden: int = 128):
            super().__init__()
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
            )
            self.pi = nn.Linear(hidden, action_dim)
            self.v = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor):
            h = self.trunk(x)
            return self.pi(h), self.v(h).squeeze(-1)


class MemoryPolicyController:
    """Unified interface (torch / no-torch). ``.decide`` returns a PolicyDecision."""

    def __init__(self, checkpoint: str | None = None, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint or os.environ.get("POLICY_CHECKPOINT", "")
        self.device = device
        self.net = None
        self.optim = None
        if _HAS_TORCH:
            self.net = _Net().to(device)
            self.optim = torch.optim.Adam(self.net.parameters(), lr=3e-4)
            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                try:
                    state = torch.load(self.checkpoint_path, map_location=device)
                    self.net.load_state_dict(state)
                except Exception:
                    pass

    # -------------------------------------------------- observation schema

    @staticmethod
    def observe(memcube) -> list[float]:
        tier_map = {"STM": [1, 0, 0], "MTM": [0, 1, 0], "LTM": [0, 0, 1],
                    "DECAY": [0, 0, 0], "RAW": [0, 0, 0]}
        tier_oh = tier_map.get(getattr(memcube.tier, "value", str(memcube.tier)), [0, 0, 0])
        return [
            memcube.importance,
            memcube.activation_strength,
            *tier_oh,
            memcube.policy_signal.utility_score,
            memcube.policy_signal.regret,
            math.log1p(memcube.replay_count),
            memcube.decay_rate,
        ]

    # ---------------------------------------------------------- decide

    def decide(self, memcube) -> PolicyDecision:
        obs = self.observe(memcube)
        if _HAS_TORCH and self.net is not None:
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.net(x)
                probs = F.softmax(logits, dim=-1)
                action = int(torch.multinomial(probs, 1).item())
                log_prob = float(torch.log(probs[0, action] + 1e-9).item())
            return PolicyDecision(
                action=PolicyAction(action),
                target_tier=self._tier_for(PolicyAction(action), memcube),
                decay_rate=self._decay_for(PolicyAction(action), memcube),
                log_prob=log_prob, value=float(value.item()),
            )
        # Rule-based fallback (no torch).
        action = self._rule_based(obs)
        return PolicyDecision(
            action=action,
            target_tier=self._tier_for(action, memcube),
            decay_rate=self._decay_for(action, memcube),
            log_prob=0.0, value=0.0,
        )

    # ---------------------------------------------------------- learn

    def learn(
        self,
        batch: list[tuple[list[float], int, float, list[float], bool, float]],
        *,
        clip: float = 0.2, epochs: int = 4,
    ) -> dict[str, float]:
        """Clipped PPO update. Each sample: (obs, action, reward, next_obs, done, old_log_prob)."""
        if not _HAS_TORCH or self.net is None or not batch:
            return {"loss": 0.0}
        obs = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_obs = torch.tensor([b[3] for b in batch], dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        old_log_probs = torch.tensor([b[5] for b in batch], dtype=torch.float32)

        with torch.no_grad():
            _, next_values = self.net(next_obs)
            targets = rewards + (1 - dones) * 0.99 * next_values

        losses: list[float] = []
        for _ in range(epochs):
            logits, values = self.net(obs)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-9)
            ratio = torch.exp(log_probs - old_log_probs)
            advantages = (targets - values.detach())
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - clip, 1 + clip) * advantages,
            ).mean()
            value_loss = F.mse_loss(values, targets)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(float(loss.item()))
        if self.checkpoint_path:
            try:
                os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
                torch.save(self.net.state_dict(), self.checkpoint_path)
            except Exception:
                pass
        return {"loss": sum(losses) / len(losses)}

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _rule_based(obs: Sequence[float]) -> PolicyAction:
        importance, activation, *_ = obs
        if importance > 0.8 and activation > 0.6:
            return PolicyAction.PROMOTE
        if activation < 0.1:
            return PolicyAction.FORGET
        if importance > 0.5:
            return PolicyAction.REINFORCE
        if importance < 0.2:
            return PolicyAction.CONSOLIDATE
        return PolicyAction.KEEP

    @staticmethod
    def _tier_for(action: PolicyAction, memcube) -> str:
        current = getattr(memcube.tier, "value", str(memcube.tier))
        if action == PolicyAction.PROMOTE:
            return {"STM": "MTM", "MTM": "LTM", "LTM": "LTM"}.get(current, current)
        if action == PolicyAction.FORGET:
            return "DECAY"
        if action == PolicyAction.CONSOLIDATE:
            return "LTM"
        return current

    @staticmethod
    def _decay_for(action: PolicyAction, memcube) -> float:
        base = memcube.decay_rate
        if action == PolicyAction.REINFORCE:
            return max(0.01, base * 0.5)
        if action == PolicyAction.FORGET:
            return min(0.4, base * 2)
        return base
