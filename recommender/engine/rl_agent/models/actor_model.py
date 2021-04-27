import torch


class ActorModel(torch.nn.Module):
    def __init__(self, K: int, SE: int, history_embedder: torch.nn.Module):
        super().__init__()
        self.K = K
        self.SE = SE
        self.history_embedder = history_embedder
        # WARNING: history_embedder is a model shared between actor and critic,
        # the .detach will be probably needed for proper training

        # TODO: layers initialization
        self.output_layer = torch.nn.Linear(None, K*SE)

    def forward(self,
                user: torch.Tensor,
                services_history: torch.Tensor,
                filters: torch.Tensor,
                search_phrase: torch.Tensor) -> torch.Tensor:
        """
        Performs forward propagation.

        Args:
            user: Embedded user content tensor of shape [UE]
            services_history: Services history tensor of shape [N, SE]
            filters: Embedded filters tensor of shape [FE]
            search_phrase: Embedded search phrase tensor of shape [SPE]

        """

        services_history = self.history_embedder(services_history)

        # TODO: implement missing forward computation
        weights = self.output_layer(None)
        weights = weights.reshape(self.K, self.SE)
        return weights
