# pylint: disable=fixme, missing-module-docstring, missing-class-docstring, invalid-name
# pylint: disable=missing-function-docstring, too-many-instance-attributes, too-many-arguments, not-callable

import itertools
from copy import deepcopy
from functools import wraps
from typing import Tuple
from time import time

import numpy as np
import torch

from recommender.engines.base.base_steps import ModelTrainingStep
from recommender.engines.panel_id_to_services_number_mapping import K_TO_PANEL_ID
from recommender.engines.rl.ml_components.actor import Actor
from recommender.engines.rl.ml_components.critic import Critic
from recommender.engines.rl.ml_components.history_embedder import MLPHistoryEmbedder

from recommender.models import Service


def log_time(to):
    def decorator_log_time(func):
        @wraps(func)
        def inner(*args, **kwargs):
            start = time()
            return_value = func(*args, **kwargs)
            end = time()
            to.append(end - start)
            return return_value

        return inner

    return decorator_log_time


class RLModelTrainingStep(ModelTrainingStep):
    def __init__(self, pipeline_config):
        super().__init__(pipeline_config)

        self.K = self.resolve_constant("K")
        SE = self.resolve_constant("SE")
        UE = self.resolve_constant("UE")
        I = len(Service.objects)  # TODO: Consider not reaching to the databse for this
        N = self.resolve_constant("N", 20)
        self.device = self.resolve_constant("device", "cpu")
        self.polyak = self.resolve_constant("polyak", 0.95)
        self.gamma = self.resolve_constant("gamma", 1.0)
        self.policy_delay = self.resolve_constant("policy_delay", 2)

        self.target_noise = self.resolve_constant("target_noise", 0.3)
        self.noise_clip = self.resolve_constant("noise_clip", 0.5)
        self.act_max = self.resolve_constant("act_max", 1.0)
        self.act_min = self.resolve_constant("act_min", -1.0)

        self.epochs = self.resolve_constant("epochs", 100)

        self.critics_mse_losses = []
        self.actor_train_durations = []
        self.critic_train_durations = []
        self.epoch_durations = []
        self.update_targets_durations = []
        self.training_duration = None

        self.actor, self.actor_optimizer = self._init_actor(
            self.K,
            UE,
            SE,
            N,
            I,
            self.resolve_constant("actor_layer_sizes", (128, 256, 128)),
            self.resolve_constant("actor_optimizer", torch.optim.Adam),
            self.resolve_constant("actor_optimizer_params", {"lr": 1e-4}),
        )
        self.critics, self.critics_optimizer, self.critics_params = self._init_critics(
            self.K,
            UE,
            SE,
            N,
            I,
            self.resolve_constant("critic_layer_sizes", (128, 256, 128)),
            self.resolve_constant("critic_optimizer", torch.optim.Adam),
            self.resolve_constant("critic_optimizer_params", {"lr": 1e-4}),
        )

        self.target_actor, self.target_critics = self._init_target_networks()

    def _init_actor(self, K, UE, SE, N, I, layer_sizes, optimizer, optimizer_params):
        actor = Actor(K, SE, UE, I, MLPHistoryEmbedder(SE, N), layer_sizes)
        actor = actor.to(self.device)
        optimizer = optimizer(actor.parameters(), **optimizer_params)

        return actor, optimizer

    def _init_critics(self, K, UE, SE, N, I, layer_sizes, optimizer, optimizer_params):
        critics = [
            Critic(K, SE, UE, I, MLPHistoryEmbedder(SE, N), layer_sizes)
            for _ in range(2)
        ]

        critics = [critic.to(self.device) for critic in critics]

        networks_params = itertools.chain.from_iterable(
            critic.parameters() for critic in critics
        )
        optimizer = optimizer(networks_params, **optimizer_params)

        return critics, optimizer, networks_params

    def _init_target_networks(self):
        networks = (self.actor, *self.critics)
        target_networks = [deepcopy(network) for network in networks]

        for target_network in target_networks:
            target_network.eval()
            for p in target_network.parameters():
                p.requires_grad = False

        return target_networks[0], target_networks[1:]

    def __call__(self, data=None) -> Tuple[torch.nn.Module, dict]:
        start_train = time()
        replay_buffer = data

        for _ in range(self.epochs):
            start_epoch = time()
            for i, batch in enumerate(replay_buffer):
                state, action, reward, next_state = self._extract_batch_tensors(batch)
                self._critics_train_step(state, action, reward, next_state)
                if i % self.policy_delay == 0:
                    self._actor_train_step(state)
                    self._update_target_networks()
            end_epoch = time()
            self.epoch_durations.append(end_epoch - start_epoch)

        end_train = time()
        self.training_duration = end_train - start_train
        return self.actor, self._create_metadata()

    def _extract_batch_tensors(self, batch):
        state = tuple(tensor.to(self.device) for tensor in batch["state"].values())
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_state = tuple(
            tensor.to(self.device) for tensor in batch["next_state"].values()
        )
        return state, action, reward, next_state

    def _critics_train_step(self, state, action, reward, next_state):
        start = time()
        with torch.no_grad():
            target_action = self._add_noise_observe(self.target_actor(next_state))
            target_action_values = torch.cat(
                [critic(next_state, target_action) for critic in self.critics], dim=1
            )
            min_action_value, _ = target_action_values.min(dim=1)
            target = reward + self.gamma * min_action_value
            target = target.unsqueeze(1)

        preds = torch.cat([critic(state, action) for critic in self.critics], dim=1)
        loss = ((preds - target) ** 2).mean(dim=0).sum()
        self.critics_mse_losses.append(loss.item())

        self.critics_optimizer.zero_grad()
        loss.backward()
        self.critics_optimizer.step()

        end = time()
        self.critic_train_durations.append(end - start)

    def _actor_train_step(self, state):
        start = time()
        for p in self.critics_params:
            p.requires_grad = False

        new_action = self.actor(state)
        action_value = self.critics[0](state, new_action)
        loss = -action_value.mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        for p in self.critics_params:
            p.requires_grad = True

        end = time()
        self.actor_train_durations.append(end - start)

    def _update_target_networks(self):
        start = time()
        networks = [self.actor, *self.critics]
        target_networks = [self.target_actor, *self.target_critics]

        with torch.no_grad():
            for network, target_network in zip(networks, target_networks):
                for p, target_p in zip(
                    network.parameters(), target_network.parameters()
                ):
                    target_p.data.mul_(self.polyak)
                    target_p.data.add_((1 - self.polyak) * p.data)
        end = time()
        self.update_targets_durations.append(end - start)

    def _add_noise_observe(self, action):
        noise = torch.randn_like(action).to(self.device)
        noise *= self.target_noise
        noise = noise.clamp(max=self.noise_clip, min=-self.noise_clip)
        noised_action = action + noise
        return noised_action.clamp(max=self.act_max, min=self.act_min)

    def _create_metadata(self):
        return {
            "critic_mse_losses": self.critics_mse_losses,
            "actor_train_durations": self.actor_train_durations,
            "mean_actor_train_duration": np.mean(self.actor_train_durations),
            "no_of_actor_training_steps": len(self.actor_train_durations),
            "critic_train_durations": self.critic_train_durations,
            "mean_critic_train_duration": np.mean(self.critic_train_durations),
            "no_of_critic_training_steps": len(self.critic_train_durations),
            "epoch_durations": self.epoch_durations,
            "mean_epoch_duration": np.mean(self.epoch_durations),
            "update_targets_durations": self.update_targets_durations,
            "mean_update_target_duration": np.mean(self.update_targets_durations),
            "training_duration": self.training_duration,
        }

    def save(self):
        version = K_TO_PANEL_ID.get(self.K)
        self.actor.save(version)
