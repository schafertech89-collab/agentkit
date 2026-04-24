"""Reflection agent — closes the meta-cognitive loop.

After N decisions have been made the reflection agent reviews them against
realised outcomes, produces a critique memcube, updates the KNN training set,
and emits a policy_signal adjustment that the RL controller consumes.

Reflection is the "AGI quality" seam: it is the place where the system
observes its own mistakes and systematically updates its priors, rather than
only propagating through parameter updates. It is modelled on the Reflexion
pattern (Shinn et al. 2023) with the addendum that the critique is durably
written into the semantic tier and graph-linked to the originating decision.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.memcube import MemoryType, Tier
from .base import Agent, AgentContext, AgentOutput


class ReflectionAgent(Agent):
    name = "reflection"
    output_tier = Tier.MTM

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        window = int(input_payload.get("window", 16))
        decisions = await ctx.read("decisions", []) or []
        outcomes = await ctx.read("outcomes", {}) or {}
        if len(decisions) < 2:
            cube = self.memcube(title="reflection (insufficient history)", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        recent = decisions[-window:]
        reviewed = []
        hits = 0
        for d in recent:
            outcome = outcomes.get(d.get("id"))
            if outcome is None:
                continue
            realized_pnl = float(outcome.get("pnl", 0.0))
            predicted = float(d.get("stance", 0.0))
            # "correct" iff sign(predicted) == sign(realized_pnl) and |pnl| meaningful
            correct = (predicted * realized_pnl) > 0 and abs(realized_pnl) > 1e-4
            if correct:
                hits += 1
            reviewed.append({
                "decision_id": d.get("id"),
                "symbol": d.get("symbol"),
                "venue": d.get("venue"),
                "predicted_stance": predicted,
                "realized_pnl": realized_pnl,
                "correct": correct,
            })

        n = max(1, len(reviewed))
        hit_rate = hits / n
        critique = (
            f"hit_rate={hit_rate:.2%} over {n} decisions. "
            f"Miss patterns: "
            + ", ".join(
                f"{r['symbol']}({r['realized_pnl']:+.4f})"
                for r in reviewed if not r["correct"]
            )[:500]
        )

        # Append labelled windows so the KNN agent can retrain on real outcomes.
        training_update = []
        for r in reviewed:
            if abs(r["realized_pnl"]) > 0.01:
                training_update.append({
                    "symbol": r["symbol"],
                    "prices": r.get("prices", []),
                    "volumes": r.get("volumes"),
                    "label": "trend_up" if r["realized_pnl"] > 0 else "trend_down",
                })
        if training_update:
            existing = await ctx.read("knn_training", []) or []
            # Bound the training set so it doesn't grow forever.
            combined = (existing + training_update)[-2000:]
            await ctx.write("knn_training", combined)

        # Emit a policy signal — positive utility for correct calls, regret for wrong ones.
        signal = {
            "hit_rate": hit_rate,
            "utility": (hit_rate - 0.5) * 2,
            "regret": max(0.0, 0.5 - hit_rate),
        }
        await ctx.publish({"type": "reflection", "signal": signal, "reviewed": reviewed})

        cube = self.memcube(
            title=f"reflection hit={hit_rate:.2%}",
            body=critique,
            payload={"hit_rate": hit_rate, "reviewed": reviewed, "signal": signal},
            memory_type=MemoryType.PLAINTEXT,
            importance=0.5 + 0.3 * abs(hit_rate - 0.5),
            tags=["reflection"],
        )
        cube.policy_signal.utility_score = signal["utility"]
        cube.policy_signal.regret = signal["regret"]

        await ctx.append(
            "reflections",
            {"hit_rate": hit_rate, "critique": critique, "timestamp": ctx.clock()},
        )

        return AgentOutput(agent=self.name, memcube=cube, stance=0.0,
                           explain=signal)
