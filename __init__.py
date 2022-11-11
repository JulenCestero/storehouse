from gym.envs.registration import register

from storehouse.policies.manual_policy import ehp

register(
    id="Storehouse-v3",
    entry_point="storehouse.environment.env_storehouse:Storehouse",
)
