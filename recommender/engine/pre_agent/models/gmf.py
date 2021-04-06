# pylint: disable=missing-function-docstring

"""Generalized Matrix Factorization"""

from torch.nn import Module, Embedding


class GMF(Module):
    """Generalized Matrix Factorization Model"""

    def __init__(self, users_max_id, services_max_id, mf_embedding_dim):
        super().__init__()
        self.mf_user_embedder = Embedding(users_max_id + 1, mf_embedding_dim)
        self.mf_service_embedder = Embedding(services_max_id + 1, mf_embedding_dim)

    def forward(self, users_ids, services_ids):
        mf_user_tensor = self.mf_user_embedder(users_ids)
        mf_service_tensor = self.mf_service_embedder(services_ids)

        output = mf_user_tensor * mf_service_tensor

        return output
