"""Learning primitives — replay buffer, RL memory policy, and trading-layer learners.

The trading-layer additions follow Sutton & Barto:

* ``td_lambda``           — TD(lambda) over per-agent stance weights with
                            eligibility traces.
* ``successor_features``  — psi(s) decoupling reward components from V(s).
* ``dyna_q_plus``         — Dyna-Q+ planning with novelty bonus.
* ``options``             — Sutton-Precup-Singh options framework.
* ``horde``               — Horde of GVF predictors.
* ``value_head``          — V_theta(s) used to bootstrap MC rollouts.

Every module is numpy-only by default; ``value_head`` opportunistically uses
torch when available (mirrors ``ppo_policy.py``).
"""

from .replay_buffer import ReplayBuffer, Trajectory, Transition
from .ppo_policy import MemoryPolicyController, PolicyAction
from .td_lambda import (
    DEFAULT_AGENTS,
    REGIMES,
    TDLambdaConfig,
    TDLambdaStanceLearner,
    feature_dim,
    featurize,
    reward_from_outcome,
)
from .successor_features import (
    DEFAULT_COMPONENTS,
    SFConfig,
    SuccessorFeatureEstimator,
)
from .dyna_q_plus import DynaQPlus, DynaQPlusConfig, stance_bucket
from .options import (
    Option,
    OptionState,
    OptionsSelector,
    OptionsSelectorConfig,
    default_macro_options,
    default_micro_options,
)
from .horde import GVF, GVFConfig, HordeLearner, default_horde
from .value_head import LinearValueHead, make_value_head

__all__ = [
    "MemoryPolicyController",
    "PolicyAction",
    "ReplayBuffer",
    "Trajectory",
    "Transition",
    # TD(lambda)
    "DEFAULT_AGENTS",
    "REGIMES",
    "TDLambdaConfig",
    "TDLambdaStanceLearner",
    "feature_dim",
    "featurize",
    "reward_from_outcome",
    # Successor features
    "DEFAULT_COMPONENTS",
    "SFConfig",
    "SuccessorFeatureEstimator",
    # Dyna-Q+
    "DynaQPlus",
    "DynaQPlusConfig",
    "stance_bucket",
    # Options
    "Option",
    "OptionState",
    "OptionsSelector",
    "OptionsSelectorConfig",
    "default_macro_options",
    "default_micro_options",
    # Horde
    "GVF",
    "GVFConfig",
    "HordeLearner",
    "default_horde",
    # Value head
    "LinearValueHead",
    "make_value_head",
]
