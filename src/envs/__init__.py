from functools import partial

from .multiagentenv import MultiAgentEnv
from smac.env import MultiAgentEnv, StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_args = kwargs.get("env_args", {})
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

