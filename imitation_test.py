import numpy as np
import seals  # needed to load "seals/" environments    # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

from imitation.algorithms.adversarial.airl import AIRL

import tempfile
from imitation.algorithms.dagger import SimpleDAggerTrainer

from imitation.algorithms import sqil

SEED = 42
rng = np.random.default_rng(SEED)
env = make_vec_env(
    "seals/CartPole-v0",
    rng=rng,
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
)


def train_expert():
    # note: use `download_expert` instead to download a pretrained, competent expert
    print("Training a expert.")
    expert = PPO(
        policy=MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(1_000)  # Note: change this to 100_000 to train a decent expert.
    return expert


def download_expert():
    print("Downloading a pretrained expert.")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    return expert


def sample_expert_transitions():
    # expert = train_expert()  # uncomment to train your own expert
    expert = download_expert()

    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    # return rollout.flatten_trajectories(rollouts)
    return rollouts

transitions = sample_expert_transitions()

# Behavioural Cloning
'''
bc_trainer = BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using Behavior Cloning")
bc_trainer.train(n_epochs=1)

reward, _ = evaluate_policy(
    bc_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=True,  # comment out to speed up
)
print(f"Reward after training: {reward}")
'''

# GAIL
'''
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.00001,
    n_epochs=1,
    seed=SEED,
)

reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

reward, _ = evaluate_policy(
    gail_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=False,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using GAIL")
gail_trainer.train(20000)

reward, _ = evaluate_policy(
    gail_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=3,
    render=False,  # comment out to speed up
)
print(f"Reward after training: {reward}")
'''

# AIRL
'''
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=16,
    learning_rate=0.0001,
    n_epochs=2,
    seed=SEED,
)

reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)

airl_trainer = AIRL(
    demonstrations=transitions,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

reward, _ = evaluate_policy(
    airl_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=False,  # comment out to speed up
)
print(f"Reward before training: {reward}")

print("Training a policy using AIRL")
airl_trainer.train(20000)

reward, _ = evaluate_policy(
    airl_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=False,  # comment out to speed up
)
print(f"Reward after training: {reward}")
'''

# DAgger
'''
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
)
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=download_expert(),
        bc_trainer=bc_trainer,
        rng=rng,
    )
    reward, _ = evaluate_policy(
        dagger_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=10,
        render=False,  # comment out to speed up
    )
    print(f"Reward before training: {reward}")
    dagger_trainer.train(8_000)

reward, _ = evaluate_policy(
    dagger_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=False,  # comment out to speed up
)
print(f"Reward after training: {reward}")
'''

# SQIL
sqil_trainer = sqil.SQIL(
    venv=env,
    demonstrations=transitions,
    policy="MlpPolicy",
)

reward, _ = evaluate_policy(
    sqil_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=False,  # comment out to speed up
)
print(f"Reward before training: {reward}")

sqil_trainer.train(total_timesteps=1_000_000)

reward, _ = evaluate_policy(
    sqil_trainer.policy,  # type: ignore[arg-type]
    env,
    n_eval_episodes=10,
    render=False,  # comment out to speed up
)
print(f"Reward after training: {reward}")