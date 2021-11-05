from torch.utils.data import Dataset

from recommender.engines.rl.ml_components.encoders.sars_encoder import STATE, USER, SERVICES_HISTORY, MASK, ACTION, \
    REWARD, NEXT_STATE


class ReplayBuffer(Dataset):
    """Replay Buffer used in Reinforcement Learning Actor Critic algorithm training."""

    def __init__(self, encoded_sarses: dict) -> None:
        """
        Create pytorch dataset out of SARSes.
        Args:
            encoded_sarses: Soutput of the SarsEncoder
        """

        self.encoded_sarses = encoded_sarses

    def __getitem__(self, index):
        return {
            STATE: {
                USER: self.encoded_sarses[STATE][USER][index],
                SERVICES_HISTORY: self.encoded_sarses[STATE][SERVICES_HISTORY][index],
                MASK: self.encoded_sarses[STATE][MASK][index],
            },
            ACTION: self.encoded_sarses[ACTION][index],
            REWARD: self.encoded_sarses[REWARD][index],
            NEXT_STATE: {
                USER: self.encoded_sarses[NEXT_STATE][USER][index],
                SERVICES_HISTORY: self.encoded_sarses[NEXT_STATE][SERVICES_HISTORY][
                    index
                ],
                MASK: self.encoded_sarses[NEXT_STATE][MASK][index],
            },
        }

    def __len__(self):
        return len(self.encoded_sarses[ACTION])