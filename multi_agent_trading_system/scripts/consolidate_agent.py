"""CLI: run a single consolidation pass. Schedule via systemd timer."""

from __future__ import annotations

import asyncio
import json
import os

from mats.governance.ssgm import SSGM
from mats.learning.ppo_policy import MemoryPolicyController
from mats.memory.consolidation import ConsolidateAgent


def main() -> None:
    vault = os.environ.get("OBSIDIAN_VAULT_PATH", "./vault")
    agent = ConsolidateAgent(
        vault_path=vault,
        ssgm=SSGM(),
        policy=MemoryPolicyController(),
        episodic_age_days=float(os.environ.get("CONSOLIDATE_AGE_DAYS", "7")),
    )
    result = asyncio.run(agent.run_once())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
