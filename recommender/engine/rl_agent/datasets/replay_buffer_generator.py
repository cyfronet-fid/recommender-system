import torch


class ReplayBufferGenerator:
    # WARNING: check if services history is generated (by SARSes generator) from accessed services AND clicked services!
    """It generates a replay buffer - dataset for the RL Agent"""

    def __init__(self, state_embedder=None, action_embedder=None):
        # TODO: implement the rest of initialization including the lazy loading
        pass

    def __call__(self) -> torch.utils.data.Dataset:
        """
        Generates a pytorch dataset.

        Returns:
            RL-Agent Dataset.

        """

        # TODO: implement
        pass