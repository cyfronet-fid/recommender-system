# pylint: disable-all
# import os
#
# import torch
# from mongoengine import connect
#
# from recommender.engines.rl.old_rl_training.simulation import simulate
# from recommender.engines.rl.old_rl_training.td3_agent import TD3Agent
# from recommender.engines.rl.old_rl_training.ddpg_agent import DDPGAgent
# from recommender.models import User, Service
#
# from ray import tune
#
# from settings import DevelopmentConfig
#
# EPISODES_N = 10
# LAST_EPISODES = EPISODES_N
# MAX_DEPTH = 10
# SERVICES_HISTORY_MAX_LEN = 20
#
#
# class AgentTrainable(tune.Trainable):
#     def setup(self, config):
#         connect(host=DevelopmentConfig.MONGODB_HOST)
#         self.config = config
#
#         # Instantiate environment and agent
#         # self.env = SyntheticMP(
#         #     N=SERVICES_HISTORY_MAX_LEN, advanced_search_data=False, max_depth=MAX_DEPTH
#         # )
#         self.env = None
#
#         UE = len(User.objects.first().dense_tensor)
#         SE = len(Service.objects.first().dense_tensor)
#         I = len(Service.objects)
#
#         self.agent = TD3Agent(
#             K=3,
#             SE=SE,
#             UE=UE,
#             I=I,
#             actor_layer_sizes=(64, 128, 256),  # (64, 128, 64),
#             critic_layer_sizes=(64, 128, 256),  # (64, 128, 64),
#             replay_buffer_max_size=1e4,
#             batch_size=64,
#             γ=1,
#             μ_θ_α=1e-5,
#             Q_Φ_α=1e-3,
#             ρ=0.95,
#             exploration=True,
#             train_after=64,
#             learning_freq=1,
#             train_steps_per_update=1,
#             # writer=writer,
#             device="cpu",
#             state_encoder=None,
#             service_selector=None,
#             max_depth=MAX_DEPTH,
#             max_steps_per_episode=self.env.interactions_per_user,
#             act_noise=0.2,
#             target_noise=0.1,
#             noise_clip=0.5,
#             policy_delay=2,
#             act_max=1,
#             act_min=-1,
#         )
#
#         self._update_agent_params(config)
#
#         if torch.cuda.is_available():
#             self.agent = self.agent.to("cuda")
#
#     def step(self):
#         simulate(
#             self.env,
#             self.agent,
#             episodes=EPISODES_N,
#             render=False,
#             max_episode_steps=None,
#             episodes_pb=False,
#             steps_pb=False,
#         )
#
#         mean_return = self.agent.evaluate(LAST_EPISODES)
#
#         return {"mean_return": mean_return}
#
#     def cleanup(self):
#         self.env.close()
#
#     def _update_agent_params(self, config):
#         for hp_name, hp_value in config.items():
#             setattr(self.agent, hp_name, hp_value)
#
#         # train_steps_per_update and learning_freq should be
#         # synchronized to keep the same execution time among trials
#         # Use old value if "learning_freq" hyper parameter isn't present in config
#         self.agent.train_steps_per_update = (
#             config.get("learning_freq") or self.agent.train_steps_per_update
#         )
#
#     def reset_config(self, new_config):
#         self._update_agent_params(new_config)
#         return True
#
#     def save_checkpoint(self, tmp_checkpoint_dir):
#         path = os.path.join(tmp_checkpoint_dir, "checkpoint")
#         self.agent.save(path, suppress_warning=True)
#         return tmp_checkpoint_dir
#
#     def load_checkpoint(self, tmp_checkpoint_dir):
#         path = os.path.join(tmp_checkpoint_dir, "checkpoint")
#
#         # PBT exploitation phase is carried out by `load_checkpoint` method so
#         # we want to load saved agent with its:
#         #   - weights,
#         #   - optimizers,
#         #   - replay buffer,
#         #   - etc
#         # but we want to use current config's hyperparams rather than loaded
#         # agent hyperparams, so we have to update them after agent loading.
#
#         self.agent = DDPGAgent.load(path)
#         if torch.cuda.is_available():
#             self.agent = self.agent.to("cuda")
#         self._update_agent_params(self.config)
