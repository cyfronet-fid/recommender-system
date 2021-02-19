# pylint: disable-all

from torch.optim import SGD
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from recommender.engine.pre_agent.models.neural_colaborative_filtering import (
    NeuralColaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engine.pre_agent.preprocessing.dataframe_to_tensor import (
    raw_dataset_to_tensors,
)
from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    create_raw_dataset,
    USERS,
    SERVICES,
    LABELS,
)
from recommender.engine.pre_agent.preprocessing.transformers import save_transformer
from recommender.engine.pre_agent.utilities.accuracy import accuracy_function
from recommender.engine.pre_agent.utilities.tensor_dict_dataset import (
    TensorDictDataset,
)
from recommender.engine.pre_agent.models.common import save_module

from tests.factories.marketplace import UserFactory, ServiceFactory


class TestNeuralColaborativeFiltering:
    def test_neural_laborative_filtering(self, mongo):
        # Populate database with artificial data
        _no_one_services = [ServiceFactory() for _ in range(20)]
        common_services = [ServiceFactory() for _ in range(5)]

        user1 = UserFactory()
        user1.accessed_services = user1.accessed_services + common_services
        user1.save()

        user2 = UserFactory()
        user2.accessed_services = user2.accessed_services + common_services
        user2.save()

        # Get data
        raw_dataset = create_raw_dataset()
        tensors, transformers = raw_dataset_to_tensors(raw_dataset)

        save_transformer(transformers[USERS], USERS)
        save_transformer(transformers[SERVICES], SERVICES)
        save_transformer(transformers[LABELS], LABELS)

        USER_FEATURES_DIM = tensors[USERS].shape[1]
        SERVICE_FEATURES_DIM = tensors[SERVICES].shape[1]
        USER_EMBEDDING_DIM = 32
        SERVICE_EMBEDDING_DIM = 64

        BATCH_SIZE = 32
        EPOCHS = 10
        LR = 0.01

        dataset = TensorDictDataset(
            {
                USERS: tensors[USERS],
                SERVICES: tensors[SERVICES],
                LABELS: tensors[LABELS],
            }
        )
        dataset_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        neural_cf_model = NeuralColaborativeFilteringModel(
            user_features_dim=USER_FEATURES_DIM,
            user_embedding_dim=USER_EMBEDDING_DIM,
            service_features_dim=SERVICE_FEATURES_DIM,
            service_embedding_dim=SERVICE_EMBEDDING_DIM,
        )

        loss_function = BCELoss()
        optimizer = SGD(neural_cf_model.parameters(), lr=LR)

        losses = []
        accuracies = []

        for epoch in range(EPOCHS):
            for batch in dataset_dl:
                labels = batch[LABELS]
                preds = neural_cf_model(batch[USERS], batch[SERVICES])

                loss = loss_function(preds, labels)
                acc = accuracy_function(preds, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                current_loss = loss.item()
                losses.append(current_loss)
                accuracies.append(acc)

        save_module(neural_cf_model, name=NEURAL_CF)
        save_transformer(transformers[USERS], USERS)
        save_transformer(transformers[SERVICES], SERVICES)
        save_transformer(transformers[LABELS], LABELS)
