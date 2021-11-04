# pylint: disable-all
#
# import pickle
# from copy import deepcopy, copy
#
# import torch
# from torch.nn import MSELoss, DataParallel
# from torch.optim import Adam
#
# from recommender.engines.rl.ml_components.actor import Actor
# from recommender.engines.rl.ml_components.critic import Critic
# from recommender.engines.rl.ml_components.history_embedder import (
#     MLPHistoryEmbedder,
# )
# from recommender.engines.rl.ml_components.reward_encoder import (
#     RewardEncoder,
# )
# from recommender.engines.rl.ml_components.state_encoder import StateEncoder
# from recommender.engines.rl.ml_components.service_selector import ServiceSelector
# from recommender.engines.rl.old_rl_training.ornstein_uhlenbeck_process import (
#     OrnsteinUhlenbeckProcess,
# )
# from recommender.engines.rl.old_rl_training.replay_buffer import ReplayBuffer
# from recommender.errors import InsufficientRecommendationSpace
# from recommender.models import State, Service
#
#
# def negative_mean_loss_function(x):
#     return -torch.mean(x)
#
#
# def set_optimizer_lr(optimizer, new_lr):
#     for g in optimizer.param_groups:
#         g["lr"] = new_lr
#
#
# class DDPGAgent:
#     def __init__(
#         self,
#         K,
#         SE,
#         UE,
#         I,
#         actor_layer_sizes=None,
#         critic_layer_sizes=None,
#         replay_buffer_max_size=1e6,
#         batch_size=128,
#         γ=0.995,
#         μ_θ_α=1e-4,
#         Q_Φ_α=1e-3,
#         ρ=0.95,
#         exploration=True,
#         noise_sigma=0.2,
#         train_after=0,
#         learning_freq=1,
#         train_steps_per_update=1,
#         writer=None,
#         device="cpu",
#         state_encoder=None,
#         service_selector=None,
#         reward_encoder=None,
#         N=20,
#         max_depth=10,
#         max_steps_per_episode=100,
#     ):
#         self.state_encoder = state_encoder or StateEncoder()
#         self.service_selector = service_selector or ServiceSelector()
#         self.reward_encoder = reward_encoder or RewardEncoder(
#             max_depth=max_depth, max_steps_per_episode=max_steps_per_episode
#         )  # TODO: what about passing reward_encoder as a parameter?? (max_depth)
#
#         # TODO: getter/setter methods
#         self.K = K
#         self.SE = SE
#         self.UE = UE
#         self.I = I
#
#         self.device = device
#
#         self.N = N  # TODO: to make it assignable, setter method handling history embedders should be implemented
#
#         # Actor
#         self.μ_θ = Actor(
#             K=K,
#             SE=SE,
#             UE=UE,
#             I=I,
#             layer_sizes=actor_layer_sizes,
#             history_embedder=MLPHistoryEmbedder(SE=SE, max_N=N),  # TODO: proper history embedder loading from database
#         ).to(device)
#
#         self.μ_θ_ℒ_function = (
#             negative_mean_loss_function  # Negative because gradient ascent
#         )
#         self._μ_θ_α = μ_θ_α
#         self.μ_θ_optimizer = Adam(self.μ_θ.parameters(), μ_θ_α)
#
#         # Critic
#         self.Q_Φ = Critic(
#             K=K,
#             SE=SE,
#             UE=UE,
#             I=I,
#             layer_sizes=critic_layer_sizes,
#             history_embedder=MLPHistoryEmbedder(SE=SE, max_N=N),  # TODO: proper history embedder loading from database
#         ).to(device)
#
#         self.Q_Φ_ℒ_function = MSELoss()
#         self._Q_Φ_α = Q_Φ_α
#         self.Q_Φ_optimizer = Adam(self.Q_Φ.parameters(), Q_Φ_α)
#
#         # Target networks
#         self.ρ = ρ
#         self.μ_θ_targ = deepcopy(self.μ_θ)
#         self.Q_Φ_targ = deepcopy(self.Q_Φ)
#         self.Q_Φ_targ.eval()
#         self.μ_θ_targ.eval()
#
#         # Replay Buffer
#         self._batch_size = batch_size
#         self._replay_buffer_max_size = replay_buffer_max_size
#         self.Ɗ = ReplayBuffer(
#             batch_size=batch_size,
#             max_size=replay_buffer_max_size,
#         )
#
#         # Ornstein Uhlenbeck Process
#         self.exploration = exploration
#         self._noise_sigma = noise_sigma
#         self._ouprocess = OrnsteinUhlenbeckProcess(K=K, SE=SE, sigma=noise_sigma)
#
#         # Other hyper-parameters
#         self.γ = γ
#
#         self.train_after = train_after
#         self.learning_freq = learning_freq
#         self.train_steps_per_update = train_steps_per_update
#
#         # Auxiliary variables
#         self._last_S = None
#         self._last_weights = None
#
#         self.steps_counter = 0
#         self.episodes_counter = 0
#
#         self.writer = writer
#
#         self.returns = []
#         self._last_return = 0
#
#         self.last_sars_valid = True
#         self.max_depth = max_depth
#         self.max_steps_per_episode = max_steps_per_episode
#
#     def act(self, S):
#         self._last_S = S
#
#         S, mask = self._prepare_state(S)
#         with torch.no_grad():
#             # self.μ_θ.eval() # TODO: maybe remove batchnorm and train/eval
#             weights = self.μ_θ(S)
#             # self.μ_θ.train()
#             if self.exploration:
#                 weights = self._add_noise(weights)
#
#             try:
#                 A = self._prepare_action(weights, mask)
#                 self.last_sars_valid = True
#             except InsufficientRecommendationSpace:
#                 A = None
#                 self.last_sars_valid = False
#
#         self._last_weights = weights
#
#         return A
#
#     def observe(self, R, S_prim, d):
#         if not self.last_sars_valid:
#             return
#
#         self._last_return += self.reward_encoder([R]).item()
#
#         if d:
#             return_val = copy(self._last_return)
#             self.returns.append(return_val)
#             self._last_return = 0
#             self._log_at_episode_end()
#             self.episodes_counter += 1
#
#         S = self._last_S
#         weights = self._last_weights
#
#         self.Ɗ << (S, weights, R, S_prim, d)
#         self.steps_counter += 1
#         # TODO: adding to the database can be performed here XD
#
#         if self._update_time():
#             for _ in range(self.train_steps_per_update):
#                 self.train_step()
#
#     def train_step(self):
#         batch = next(iter(self.Ɗ))
#         batch = self._cast_batch_to_device(batch)
#
#         self._log_inputs(batch)
#
#         self._critic_train_step(batch)
#         self._actor_train_step(batch)
#         self._target_nets_train_step()
#
#     def evaluate(self, last_n_episodes=100):
#         mean_return = torch.mean(torch.Tensor(self.returns[-last_n_episodes:])).item()
#         return mean_return
#
#     def _critic_train_step(self, batch):
#         S, A, R, S_prim, d = batch
#
#         # Critic loss calculation
#         with torch.no_grad():
#             y = R + self.γ * (1 - d) * self.Q_Φ_targ(
#                 S_prim, self.μ_θ_targ(S_prim)
#             ).squeeze(1)
#             y = y.reshape(-1, 1)
#         y_pred = self.Q_Φ(S, A)
#         Q_Φ_ℒ = self.Q_Φ_ℒ_function(y_pred, y)
#         self._log_scalar("Loss/Q_Φ_ℒ", Q_Φ_ℒ)
#
#         # Weights update
#         self.Q_Φ_optimizer.zero_grad()
#         Q_Φ_ℒ.backward()
#         # clip_grad_value_(self.Q_Φ.parameters(), clip_value=0.5)
#         # clip_grad_norm_(self.Q_Φ.parameters(), max_norm=1.0, norm_type=2.0)
#         self.Q_Φ_optimizer.step()
#
#     def _actor_train_step(self, batch):
#         S, _, _, _, _ = batch
#
#         # Actor loss calculation
#         A = self.μ_θ(S)
#         action_value = self.Q_Φ(S, A)
#         μ_θ_ℒ = self.μ_θ_ℒ_function(action_value)
#         self._log_scalar("Loss/μ_θ_ℒ", μ_θ_ℒ)
#
#         # Actor weights update
#         self.μ_θ_optimizer.zero_grad()
#         μ_θ_ℒ.backward()
#         # clip_grad_value_(self.μ_θ.parameters(), clip_value=0.5)
#         # clip_grad_norm_(self.μ_θ.parameters(), max_norm=1.0, norm_type=2.0)
#         self.μ_θ_optimizer.step()
#
#     def _target_nets_train_step(self):
#         with torch.no_grad():
#             for Φ, Φ_targ in zip(self.Q_Φ.parameters(), self.Q_Φ_targ.parameters()):
#                 Φ_targ.data.copy_(self.ρ * Φ_targ.data + (1 - self.ρ) * Φ.data)
#             for θ, θ_targ in zip(self.μ_θ.parameters(), self.μ_θ_targ.parameters()):
#                 θ_targ.data.copy_(self.ρ * θ_targ.data + (1 - self.ρ) * θ.data)
#
#     # DATA TRANSFORMING AND UTILITIES
#     def _prepare_state(self, S: State):
#         state_batch = self.state_encoder([S])
#         state_batch = [t.to(self.device) for t in state_batch]
#         mask_batch = state_batch[-1]
#         return state_batch, mask_batch
#
#     def _prepare_action(self, weights, mask):
#         # TODO: device handling?
#         services_ids = self.service_selector(
#             K=self.K, weights=weights.cpu().squeeze(), mask=mask.cpu().squeeze()
#         )
#         action = Service.objects(id__in=services_ids)
#         return action
#
#     def _add_noise(self, action):
#         noise = torch.Tensor(self._ouprocess.sample()).to(self.device)
#         noised_action = action + noise
#         return noised_action
#
#     def _cast_batch_to_device(self, batch):
#         casted_batch = []
#         for part in batch:
#             if isinstance(part, torch.Tensor):
#                 casted_batch.append(part.to(self.device))
#             elif isinstance(part, tuple):
#                 casted_batch.append(self._cast_batch_to_device(part))
#             else:
#                 raise Exception("invalid batch part")
#
#         casted_batch = tuple(casted_batch)
#         return casted_batch
#
#     def _update_time(self):
#         flag = (
#             self.steps_counter >= self.train_after
#             and self.steps_counter % self.learning_freq == 0
#         )
#         return flag
#
#     # LOGGING
#     def _log_scalar(self, name, scalar, counter=None):
#         if self.writer is None:
#             return
#
#         counter = counter or self.steps_counter
#         self.writer.add_scalar(name, scalar, counter)
#
#     def _log_nets_weights_histograms(self):
#         if self.writer is None:
#             return
#
#         nets = {
#             "μ_θ": self.μ_θ,
#             "Q_Φ": self.Q_Φ,
#             "μ_θ_targ": self.μ_θ_targ,
#             "Q_Φ_targ": self.Q_Φ_targ,
#         }
#         for net_name, net in nets.items():
#             for param_name, param in net.named_parameters():
#                 self.writer.add_histogram(
#                     f"{net_name}/{param_name}", param.data, self.episodes_counter
#                 )
#
#     def _log_inputs(self, batch):
#         if self.writer is None:
#             return
#
#         S, A, R, S_prim, d = batch
#
#         self.writer.add_scalars(
#             "Batch/S",
#             {
#                 "user norm": torch.norm(S[0]),
#                 "services_history norm": torch.norm(S[1]),
#                 "mask norm": torch.norm(S[2]),
#             },
#             self.steps_counter,
#         )
#
#         self.writer.add_scalar("Batch/A", torch.norm(A), self.steps_counter)
#
#         self.writer.add_scalar("Batch/R", torch.norm(R), self.steps_counter)
#
#         self.writer.add_scalars(
#             "Batch/S_prim",
#             {
#                 "user norm": torch.norm(S_prim[0]),
#                 "services_history norm": torch.norm(S_prim[1]),
#                 "mask norm": torch.norm(S_prim[2]),
#             },
#             self.steps_counter,
#         )
#
#     def _flush_logs(self):
#         if self.writer is not None:
#             self.writer.flush()
#
#     def _log_at_episode_end(self):
#         self._log_nets_weights_histograms()
#         self._log_scalar("Return", self.returns[-1], self.episodes_counter)
#         self._log_scalar("Replay Buffer size", len(self.Ɗ.buffer))
#         self._flush_logs()
#
#     # COPYING AND SAVING
#
#     def to(self, device):
#         writer = self.writer
#         self.writer = None
#
#         new = deepcopy(self)
#         self.writer = writer
#         new.writer = writer
#
#         new.device = device
#
#         if new.device == "cuda":
#             if torch.cuda.is_available():
#                 if torch.cuda.device_count() > 1:
#                     new.μ_θ = DataParallel(new.μ_θ)
#                     new.Q_Φ = DataParallel(new.Q_Φ)
#             else:
#                 new.device = "cpu"
#
#         new.μ_θ_targ = deepcopy(new.μ_θ)
#         new.Q_Φ_targ = deepcopy(new.Q_Φ)
#         new.μ_θ_targ.eval()
#         new.Q_Φ_targ.eval()
#
#         new.μ_θ = new.μ_θ.to(new.device)
#         new.Q_Φ = new.Q_Φ.to(new.device)
#         new.μ_θ_targ = new.μ_θ_targ.to(new.device)
#         new.Q_Φ_targ = new.Q_Φ_targ.to(new.device)
#
#         # Optimizers has to be re-instantiated after moving nets to selected device
#         new.μ_θ_optimizer = new.μ_θ_optimizer.__class__(new.μ_θ.parameters(), new.μ_θ_α)
#         new.Q_Φ_optimizer = new.Q_Φ_optimizer.__class__(new.μ_θ.parameters(), new.μ_θ_α)
#
#         return new
#
#     def save(self, file_path, suppress_warning=False):
#         writer = self.writer
#         self.writer = None
#         self._last_S = None  # WARNING:
#         with open(file_path, "wb") as file:
#             pickle.dump(self, file)
#         self.writer = writer
#         if not suppress_warning:
#             print(
#                 "Agent saved successfully! (agent.writer object can't be saved so"
#                 " this field has been set to `None` in saved agent, but hasn't been changed in current agent object). _last_S couldn't be saved also."
#             )
#
#     @classmethod
#     def load(cls, file_path):
#         with open(file_path, "rb") as file:
#             return pickle.load(file)
#
#     # GETTERS AND SETTERS
#     @property
#     def batch_size(self):
#         return self._batch_size
#
#     @batch_size.setter
#     def batch_size(self, new_batch_size):
#         self._batch_size = new_batch_size
#         self.Ɗ.batch_size = new_batch_size
#
#     @property
#     def noise_sigma(self):
#         return self._noise_sigma
#
#     @noise_sigma.setter
#     def noise_sigma(self, new_value):
#         self._noise_sigma = new_value
#         self._ouprocess.sigma = new_value
#
#     @property
#     def replay_buffer_max_size(self):
#         return self._replay_buffer_max_size
#
#     @replay_buffer_max_size.setter
#     def replay_buffer_max_size(self, new_value):
#         new_value = int(new_value)
#         self._replay_buffer_max_size = new_value
#         self.Ɗ._max_size = new_value
#
#     @property
#     def μ_θ_α(self):
#         return self._μ_θ_α
#
#     @μ_θ_α.setter
#     def μ_θ_α(self, new_value):
#         self._μ_θ_α = new_value
#         set_optimizer_lr(self.μ_θ_optimizer, new_value)
#
#     @property
#     def Q_Φ_α(self):
#         return self._Q_Φ_α
#
#     @Q_Φ_α.setter
#     def Q_Φ_α(self, new_value):
#         self._Q_Φ_α = new_value
#         set_optimizer_lr(self.Q_Φ_optimizer, new_value)
