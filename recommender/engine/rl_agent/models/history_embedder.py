import torch


class HistoryEmbedder(torch.nn.Module):
    """
    Model used for transforming services history (list of services tensors
     in temporal order) into history tensor. It should be used and trained inside actor and critic.
    """

    def __init__(self):
        super().__init__()
        self.rnn = None

        # TODO: implement the rest of the initialization (layers)
        pass

    def forward(self, services_history: torch.Tensor) -> torch.Tensor:
        """
        RNN is used for reducing history N dimension.

        Args:
            services_history: user clicked services history tensor of shape
             [N, SE] where N is the history length and SE is service content tensor embedding dim

        Returns:
            Clicked services history embedding tensor of shape [SE]
        """

        # TODO: implement forward computation (using RNN)
        pass
