"""MATS orchestrator.

Coordinates the full agent fleet. Each tick:
  1. Data-collection agents populate the SCS (fundamentals, regime, MC paths)
  2. Analysis agents emit stances (technical, fundamental, quant, MC, KNN)
  3. Risk agent fuses stances, generates per-venue SizedOrders
  4. Execution agent submits orders simultaneously across venues
  5. Allora publisher sends signed predictions to configured topics
  6. Reward sweeper claims rewards + swaps to Venice DIEM
  7. Reflection agent runs every N ticks, updating the policy + KNN training

Every MemCube produced flows through the memory policy controller which
decides which tier it lands in.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..agents import (
    AlloraPublisherAgent,
    ExecutionAgent,
    FundamentalAgent,
    FundamentalDataAgent,
    KNNRegimeAgent,
    MonteCarloAgent,
    QuantitativeAgent,
    ReflectionAgent,
    RewardSweeperAgent,
    RiskAgent,
    TechnicalAgent,
)
from ..agents.base import AgentContext, AgentOutput
from ..connectors import (
    AlloraConnector,
    BaseChainConnector,
    CoinbaseConnector,
    PredictBaseConnector,
    VeniceConnector,
    WasabiConnector,
)
from ..governance.ssgm import SSGM
from ..learning.ppo_policy import MemoryPolicyController, PolicyAction
from ..learning.replay_buffer import ReplayBuffer, Transition
from ..memory.vault_writer import VaultWriter
from .hooks import ConduitHooks, DEFAULT_HOOKS
from .memcube import MemCube, Tier
from .scs import SCS, build_scs


log = logging.getLogger("mats.orchestrator")


@dataclass
class OrchestratorConfig:
    tier: int = 1
    session_id: str = "default"
    universe: list[tuple[str, str]] = field(default_factory=lambda: [
        ("ETH-USD", "coinbase"),
        ("BTC-USD", "coinbase"),
        ("SOL-USD", "coinbase"),
    ])
    wasabi_symbols: list[str] = field(default_factory=list)
    predictbase_markets: list[str] = field(default_factory=list)
    allora_topics: list[tuple[int, str]] = field(default_factory=list)
    reward_tokens: list[str] = field(default_factory=list)
    tick_interval_sec: float = 60.0
    reflection_every: int = 10
    sweep_every: int = 20
    account_equity: float = 10_000.0
    max_leverage: float = 3.0
    vault_path: str = "./vault"
    scs_backend: str = "memory"
    scs_kwargs: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig | None = None,
                 hooks: ConduitHooks | None = None) -> None:
        self.cfg = cfg or OrchestratorConfig()
        self.hooks = hooks or DEFAULT_HOOKS
        self.scs: SCS = build_scs(self.cfg.scs_backend, **self.cfg.scs_kwargs)
        self.ssgm = SSGM()
        self.writer = VaultWriter(self.cfg.vault_path, ssgm=self.ssgm)
        self.policy = MemoryPolicyController()
        self.replay = ReplayBuffer()
        self.tick_count = 0
        self._stop = False
        self.connectors = self._build_connectors()
        self._agents = self._build_agents()

    # --------------------------------------------------------- setup

    def _build_connectors(self) -> dict[str, Any]:
        base = BaseChainConnector()
        return {
            "coinbase": CoinbaseConnector(),
            "wasabi": WasabiConnector(chain=base),
            "predictbase": PredictBaseConnector(chain=base),
            "base": base,
            "allora": AlloraConnector(),
            "venice": VeniceConnector(chain=base),
        }

    def _build_agents(self) -> dict[str, Any]:
        return {
            "fundamental_data": FundamentalDataAgent(hooks=self.hooks),
            "technical": TechnicalAgent(hooks=self.hooks),
            "fundamental": FundamentalAgent(hooks=self.hooks),
            "quant": QuantitativeAgent(hooks=self.hooks),
            "monte_carlo": MonteCarloAgent(hooks=self.hooks),
            "knn": KNNRegimeAgent(),
            "risk": RiskAgent(hooks=self.hooks),
            "execution": ExecutionAgent(hooks=self.hooks),
            "allora": AlloraPublisherAgent(hooks=self.hooks),
            "reflection": ReflectionAgent(hooks=self.hooks),
            "sweep": RewardSweeperAgent(hooks=self.hooks),
        }

    # --------------------------------------------------------- loop

    async def run(self) -> None:
        log.info("orchestrator tier=%d session=%s", self.cfg.tier, self.cfg.session_id)
        while not self._stop:
            try:
                await self.tick()
            except Exception as exc:  # noqa: BLE001
                log.exception("tick failed: %s", exc)
                await self.hooks.emit("error", exc)
            await asyncio.sleep(self.cfg.tick_interval_sec)

    def stop(self) -> None:
        self._stop = True

    async def tick(self) -> dict[str, Any]:
        ctx = AgentContext(
            session_id=self.cfg.session_id,
            scs=self.scs, connectors=self.connectors,
            hooks=self.hooks, ssgm=self.ssgm,
        )
        tick_id = self.tick_count
        self.tick_count += 1
        summary: dict[str, Any] = {"tick": tick_id, "symbols": {}}

        # 1. Collect fundamentals in parallel
        await asyncio.gather(*[
            self._run("fundamental_data", ctx, {"symbol": sym})
            for sym, _ in self.cfg.universe
        ])

        # 2. Analysis per symbol in parallel
        all_outputs: list[AgentOutput] = []
        for symbol, venue in self.cfg.universe:
            outs = await asyncio.gather(
                self._run("technical", ctx, {"symbol": symbol, "venue": venue}),
                self._run("fundamental", ctx, {"symbol": symbol}),
                self._run("quant", ctx, {"symbol": symbol, "venue": venue,
                                          "universe": [s for s, _ in self.cfg.universe]}),
                self._run("monte_carlo", ctx, {"symbol": symbol, "venue": venue,
                                                "horizon_days": 1.0, "target_return": 0.01}),
                self._run("knn", ctx, {"symbol": symbol, "venue": venue}),
            )
            all_outputs.extend(outs)
            # 3. Risk / sizing (runs after analysis finishes)
            risk_out = await self._run("risk", ctx, {
                "symbol": symbol, "venue": venue,
                "account_equity": self.cfg.account_equity,
                "max_leverage": self.cfg.max_leverage,
            })
            all_outputs.append(risk_out)
            summary["symbols"][symbol] = {
                "stance": risk_out.stance,
                "actions": risk_out.actions,
            }

        # 4. Execution
        combined_actions = [a for out in all_outputs for a in out.actions]
        exec_out = await self._run("execution", ctx, {"actions": combined_actions})
        all_outputs.append(exec_out)

        # 5. Allora publication
        predictions = []
        for topic_id, symbol in self.cfg.allora_topics:
            mc = await ctx.read(f"mc:{symbol}", {}) or {}
            predictions.append({
                "topic_id": topic_id, "symbol": symbol,
                "value": mc.get("mean") or mc.get("s0") or 0.0,
                "confidence": summary["symbols"].get(symbol, {}).get("stance", 0.0),
            })
        if predictions:
            allora_out = await self._run("allora", ctx, {"predictions": predictions})
            all_outputs.append(allora_out)

        # 6. Reflection & sweep on schedule
        if self.tick_count % max(1, self.cfg.reflection_every) == 0:
            reflection_out = await self._run("reflection", ctx, {"window": 32})
            all_outputs.append(reflection_out)
        if self.tick_count % max(1, self.cfg.sweep_every) == 0:
            sweep_out = await self._run("sweep", ctx, {
                "reward_tokens": self.cfg.reward_tokens,
                "resolved_markets": [],
            })
            all_outputs.append(sweep_out)

        # 7. Route every cube through the memory policy + writer
        for out in all_outputs:
            self._route_memcube(out.memcube)

        return summary

    # --------------------------------------------------------- plumbing

    async def _run(self, agent_key: str, ctx: AgentContext, payload: dict[str, Any]) -> AgentOutput:
        agent = self._agents[agent_key]
        out = await agent.run_safely(ctx, payload)
        await self._publish_trace(agent_key, payload, out)
        await self._collect_replay(agent_key, payload, out)
        return out

    async def _publish_trace(self, agent: str, payload: dict, out: AgentOutput) -> None:
        msg = {
            "type": "agent_trace",
            "tick": self.tick_count,
            "agent": agent,
            "payload": {k: v for k, v in payload.items() if k != "actions"},
            "stance": out.stance,
            "explain": out.explain,
            "ts": time.time(),
        }
        await self.scs.publish(f"scs:{{{self.cfg.session_id}}}:events", msg)

    async def _collect_replay(self, agent: str, payload: dict, out: AgentOutput) -> None:
        obs = self.policy.observe(out.memcube)
        action = self.policy.decide(out.memcube).action.value
        reward = float(out.memcube.policy_signal.utility_score)
        t = Transition(
            obs=obs, action=action, reward=reward,
            next_obs=obs, done=False,
            metadata={"agent": agent, "tick": self.tick_count},
        )
        await self.replay.add(t)

    def _route_memcube(self, cube: MemCube) -> None:
        decision = self.policy.decide(cube)
        if decision.action == PolicyAction.PROMOTE:
            cube.promote(Tier(decision.target_tier) if decision.target_tier in Tier._value2member_map_
                         else cube.tier)
        elif decision.action == PolicyAction.FORGET:
            cube.tier = Tier.DECAY
        elif decision.action == PolicyAction.REINFORCE:
            cube.replay(reward=0.05)
        elif decision.action == PolicyAction.CONSOLIDATE:
            cube.tier = Tier.LTM
        try:
            self.writer.write(cube)
        except Exception as exc:  # noqa: BLE001
            log.warning("vault write failed: %s", exc)
