import numpy as np
from mongoengine import connect
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining


# Ray Tune verbosity modes
from definitions import LOG_DIR
from recommender.engine.agents.rl_agent.training.agent_trainable import AgentTrainable
from recommender.engine.agents.rl_agent.training.hyperparameters import get_hp_staff
from settings import DevelopmentConfig

SILENT = 0
ONLY_STATUS_UPDATES = 1
STATUS_AND_BRIEF_TRIAL_RESULTS = 2
STATUS_AND_DETAILED_TRIAL_RESULTS = 3

if __name__ == "__main__":
    connection = connect(host=DevelopmentConfig.MONGODB_HOST)

    cpu_per_trial = 1
    gpu_per_trial = 0

    TRIALS_N = 10

    config, hyperparam_mutations, explore = get_hp_staff()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=1,
        hyperparam_mutations=hyperparam_mutations,
        custom_explore_fn=explore,
        resample_probability=0.25,
        synch=True,
    )

    analysis = tune.run(
        AgentTrainable,
        config=config,
        scheduler=pbt,
        metric="mean_return",
        mode="max",
        fail_fast=True,
        stop={"training_iteration": 400, "mean_return": 1},
        num_samples=TRIALS_N,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        local_dir=LOG_DIR,
        verbose=ONLY_STATUS_UPDATES,
        reuse_actors=True,
        queue_trials=True,
    )

    print("Best config: ", analysis.get_best_config(metric="mean_return", mode="max"))

    df = analysis.dataframe()
    print(df)
