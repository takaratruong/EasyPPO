from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

register(
    id="Skeleton",
    entry_point="env.skeleton:Skeleton",
)

register(
    id="Skeleton_AMP",
    entry_point="env.skeleton_amp:Skeleton_AMP",
)

register(
    id="SkeletonRender",
    entry_point="env.skel_render:SkeletonRender",
)

