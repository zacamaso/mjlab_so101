import gymnasium as gym


gym.register(
    id="Mjlab-PPSimple",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PPSimpleEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PPSimpleRunnerCfg",
    },
)


gym.register(
    id="Mjlab-PPSimple-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:PPSimpleEnvCfgPlay",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:PPSimpleRunnerCfg",
    },
)




