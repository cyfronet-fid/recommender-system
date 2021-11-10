# pylint: disable-all
#
# from ray import tune
#
#
# def get_hp_staff():
#     config = {
#         # "replay_buffer_max_size": tune.grid_search([10 ** x for x in range(3, 7)]),
#         # "batch_size": tune.grid_search([2 ** x for x in range(4, 10)]),
#         # "learning_freq": tune.grid_search([2 ** x for x in range(0, 5)]),
#         "γ": tune.uniform(0.9, 1),
#         "μ_θ_α": tune.loguniform(1e-4, 1e-1),
#         "Q_Φ_α": tune.loguniform(1e-4, 1e-1),
#         "ρ": tune.loguniform(0.5, 0.95),
#         "act_noise": tune.uniform(0.1, 0.5),
#         "target_noise": tune.uniform(0.1, 0.5),
#         "noise_clip": tune.uniform(0.1, 0.5),
#         "policy_delay": tune.choice([2, 3, 4]),
#     }
#
#     hyperparam_mutations = {
#         # "replay_buffer_max_size": tune.choice([10 ** x for x in range(3, 7)]),
#         # "batch_size": tune.choice([2 ** x for x in range(4, 10)]),
#         # "learning_freq": tune.choice([2 ** x for x in range(0, 5)]),
#         "γ": tune.uniform(0.9, 1),
#         "μ_θ_α": tune.loguniform(1e-6, 1e-1),
#         "Q_Φ_α": tune.loguniform(1e-6, 1e-1),
#         "ρ": tune.loguniform(0.5, 0.95),
#         "act_noise": tune.uniform(0.1, 0.5),
#         "target_noise": tune.uniform(0.1, 0.5),
#         "noise_clip": tune.uniform(0.1, 0.5),
#         "policy_delay": tune.choice([2, 3, 4]),
#     }
#
#     # Utility function
#     def clip_limits(config, key, lower, upper):
#         if config[key] < lower:
#             config[key] = lower
#
#         if config[key] > upper:
#             config[key] = upper
#
#     # Postprocess the perturbed config to ensure it's still valid
#     def explore(config):
#         # clip_limits(config, "replay_buffer_max_size", 10**3, 10**6)
#         # clip_limits(config, "batch_size", 16, 512)
#         # clip_limits(config, "learning_freq", 1, 16)
#         clip_limits(config, "γ", 0.9, 1)
#         clip_limits(config, "μ_θ_α", 1e-6, 1e-1)
#         clip_limits(config, "Q_Φ_α", 1e-6, 1e-1)
#         clip_limits(config, "ρ", 0.5, 0.95)
#         clip_limits(config, "act_noise", 0.1, 0.5)
#         clip_limits(config, "target_noise", 0.1, 0.5)
#         clip_limits(config, "noise_clip", 0.1, 0.5)
#         clip_limits(config, "policy_delay", 2, 4)
#
#         return config
#
#     return config, hyperparam_mutations, explore
