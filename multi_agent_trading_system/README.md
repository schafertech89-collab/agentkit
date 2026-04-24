# Multi-Agent Trading System (MATS)

A provider-agnostic, MCP-first multi-agent trading system that trades simultaneously on **Coinbase**,
**PredictBase**, **Base**, and **Wasabi Protocol on Base**, publishes predictions to **Allora**, and
routes any rewards to a Base wallet where they are swapped for **Venice.ai DIEM**.

## Design lineage

MATS is the trading specialization of the five-tier Claude+Ollama+Obsidian+MCP+Perchance architecture:

1. **Conduit** — flat vault, BM25, single dumb relay
2. **PARA-Semantic Router** — Zettel vault, local embeddings, LiteLLM router
3. **Tiered Mnemonic Bus** — episodic/semantic/procedural isolation via CA-MCP shared context store
4. **Graph-Cognized Substrate** — Kuzu/LightRAG knowledge graph + A2A bridge
5. **Neurocognitive Memory OS** — MemCubes, SSGM governance, RL-governed consolidation,
   Virtual MCP gateway with Lua routing

Every tier is independently deployable. MATS layers all five so that a fresh install is Conduit
out-of-the-box and can be progressively hardened without rewriting any component.

## What it does

MATS runs a fleet of cooperating agents that:

- Pull order-book, on-chain, news and prediction-market data
- Run **Monte Carlo** path simulations, **KNN** regime classification, a **reflection** loop,
  quantitative/technical/fundamental scoring
- Compute Kelly-sized, Sharpe-aware risk/reward positions
- Execute simultaneously across Coinbase (spot + perps), Wasabi (perps on Base), PredictBase
  (on-chain prediction markets), and raw Base DEX swaps
- Publish every prediction as a signed worker submission to Allora
- Monitor Allora/PredictBase payouts, sweep them to a designated Base wallet, and route swaps into
  Venice.ai DIEM

All of this happens behind a Virtual MCP gateway, with SSGM audit logging on every write and an
RL-trained memory policy deciding what moves between STM / MTM / LTM tiers.

## Quick start (Tier 1 — Conduit)

```bash
cd multi_agent_trading_system
cp .env.example .env           # fill in keys
docker compose -f compose/tier1.yml up -d
python -m mats.scripts.run_orchestrator --tier 1
```

Visit the bundled Perchance panel (see `perchance/generator.js`) or hit the relay directly:

```bash
curl -N -H "X-Conduit-Secret: $CONDUIT_SHARED_SECRET" \
  -d '{"prompt": "scan eth markets"}' \
  http://localhost:8787/chat
```

## Tier map

| Tier | Compose file       | Extras added                                                                  |
| ---- | ------------------ | ----------------------------------------------------------------------------- |
| 1    | `compose/tier1.yml`| Ollama + LiteLLM + Obsidian REST + conduit-relay + bm25 search                |
| 2    | `compose/tier2.yml`| + Chroma, PARA router MCP, Cloudflare Worker edge relay                       |
| 3    | `compose/tier3.yml`| + Redis SCS, episodic/semantic/procedural MCPs, consolidate-agent timer       |
| 4    | `compose/tier4.yml`| + Kuzu graph, LightRAG hybrid retrieval, A2A graph-builder                    |
| 5    | `compose/tier5.yml`| + Keycloak, ToolHive VirtualMCP, Lua routing, TEI, MemOS, SSGM governance     |

See `docs/ARCHITECTURE.md` for detailed per-tier component breakdown, ports, and failure modes.

## Safety

This system places real orders on real exchanges. It is **off by default** — every connector boots
in *paper-trading* mode and only graduates to live execution when `MATS_LIVE=1` is set in `.env`
AND the governance audit chain's genesis hash matches. See `docs/SAFETY.md`.

## License

MIT (see repository root `LICENSE.md`).
