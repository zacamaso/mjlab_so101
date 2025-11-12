from dataclasses import dataclass, field

from mjlab.rl.config import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@dataclass
class PPSimpleRunnerCfg(RslRlOnPolicyRunnerCfg):
    experiment_name: str = "pp_simple"
    num_envs: int = 1024
    clip_actions: None = None
    policy: RslRlPpoActorCriticCfg = field(
        default_factory=lambda: RslRlPpoActorCriticCfg(
            init_noise_std=1.0,  # Reduced from 1.0 to prevent ballooning
            noise_std_type="scalar",  # Scalar prevents unbounded growth vs "log"
            actor_hidden_dims=(512, 256, 256, 128),
            critic_hidden_dims=(512, 256, 256, 128),
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            activation="elu",
        )
    )
    algorithm: RslRlPpoAlgorithmCfg = field(
        default_factory=lambda: RslRlPpoAlgorithmCfg(
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=1e-3,  # Reduced from 1e-3 for more stable training
            entropy_coef=0.25,  # Increased from 0.2 to 0.5 to add exploration
            clip_param=0.2,
            max_grad_norm=3.0,  # Tighter gradient clipping for stability. was 2.2
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            
        )
    )

