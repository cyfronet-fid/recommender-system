# pylint: disable-all
import string
import random

import torch
from dotenv import load_dotenv

from definitions import LOG_DIR
from recommender.engines.rl.old_rl_training.td3_agent import TD3Agent

load_dotenv()
from mongoengine import connect, disconnect

from settings import DevelopmentConfig

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import trange, tqdm

from recommender.models import User, Service
from recommender.engines.rl.ml_components.synthetic_dataset.env import SyntheticMP


def simulate(
    env,
    agent,
    episodes,
    render=True,
    max_episode_steps=None,
    episodes_pb=True,
    steps_pb=True,
):
    for _ in trange(episodes, disable=(not episodes_pb)):
        S = env.reset()
        episode_steps = 0

        with tqdm(total=env.interactions_per_user, disable=(not steps_pb)) as pbar:
            while True:
                if render:
                    env.render()

                A = agent.act(S)
                S_prim, R, d, _ = env.step(A)

                if max_episode_steps is not None and episode_steps >= max_episode_steps:
                    d = True
                agent.observe(R, S_prim, d)

                if S_prim is not None:
                    S = S_prim
                    episode_steps += 1
                    pbar.update(1)
                if d:
                    break
    env.close()


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


if __name__ == "__main__":
    disconnect()
    connect(host=DevelopmentConfig.MONGODB_HOST)

    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)

    # real_logdir = LOG_DIR + "/" + get_random_string(20)
    real_logdir = LOG_DIR

    writer = SummaryWriter(log_dir=real_logdir)

    MAX_DEPTH = 1
    env = SyntheticMP(N=20, advanced_search_data=False, max_depth=MAX_DEPTH)

    UE = len(User.objects.first().dense_tensor)
    SE = len(Service.objects.first().dense_tensor)
    I = len(Service.objects)

    training_agent = TD3Agent(
        K=3,
        SE=SE,
        UE=UE,
        I=I,
        actor_layer_sizes=(64, 128, 256),  # (64, 128, 64),
        critic_layer_sizes=(64, 128, 256),  # (64, 128, 64),
        replay_buffer_max_size=1e6,
        batch_size=64,
        γ=1,
        μ_θ_α=1e-5,
        Q_Φ_α=1e-3,
        ρ=0.95,
        exploration=True,
        train_after=64,
        learning_freq=1,
        train_steps_per_update=1,
        writer=writer,
        device="cuda",
        state_encoder=None,
        service_selector=None,
        max_depth=MAX_DEPTH,
        max_steps_per_episode=env.interactions_per_user,
        act_noise=0.4,
        target_noise=0.25,
        noise_clip=0.5,
        policy_delay=2,
        act_max=1,
        act_min=-1,
    )

    simulate(
        env,
        training_agent,
        episodes=400,
        render=False,
        max_episode_steps=None,
    )

    training_agent.exploration = False

    simulate(
        env,
        training_agent,
        episodes=100,
        render=False,
        max_episode_steps=None,
    )
