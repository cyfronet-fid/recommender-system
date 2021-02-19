# pylint: disable=redefined-outer-name, redefined-builtin, missing-function-docstring, no-member
# pylint: disable=too-many-locals, invalid-name, too-many-statements

"""This is example of training and inferencing
 using Neural Colaborative Filtering model.
 """

import matplotlib.pyplot as plt

from torch.optim import SGD
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from recommender.engine.pre_agent.models.neural_colaborative_filtering import (
    NeuralColaborativeFilteringModel,
    NEURAL_CF,
)
from recommender.engine.pre_agent.preprocessing.dataframe_to_tensor import (
    raw_dataset_to_tensors,
    calculate_tensors_for_users_and_services,
)
from recommender.engine.pre_agent.preprocessing.mongo_to_dataframe import (
    create_raw_dataset,
    USERS,
    SERVICES,
    LABELS,
    calculate_dfs_for_users_and_services,
)
from recommender.engine.pre_agent.utilities.accuracy import accuracy_function
from recommender.engine.pre_agent.utilities.tensor_dict_dataset import (
    TensorDictDataset,
)
from recommender.engine.pre_agent.models.common import save_module

from recommender.extensions import celery


@celery.task
def pre_agent_training():
    # Get data
    print("Generating raw dataset...")
    raw_dataset = create_raw_dataset()
    print("Raw dataset generated!\n")

    print("Transforming raw dataset to tensors...")
    tensors, _ = raw_dataset_to_tensors(raw_dataset)
    print("Raw dataset transformed to tensors!\n")

    USER_FEATURES_DIM = tensors[USERS].shape[1]
    SERVICE_FEATURES_DIM = tensors[SERVICES].shape[1]
    USER_EMBEDDING_DIM = 32
    SERVICE_EMBEDDING_DIM = 64

    BATCH_SIZE = 32
    EPOCHS = 4000
    LR = 0.01

    print("Creating dataset...")
    dataset = TensorDictDataset(tensors)
    dataset_dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataset created successfully!\n")

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

    neural_cf_model.train()

    print("Training...")
    for epoch in range(EPOCHS):
        with tqdm(dataset_dl, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
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
                tepoch.set_postfix(loss=loss.item(), accuracy=acc)
    print("\nModel trained successfully!\n")

    plt.plot([float(loss) for loss in losses])
    plt.plot([float(acc) for acc in accuracies])
    plt.show()

    # Transformers are save automatically so only model should be saved
    # manually:
    print("Model saving...")
    save_module(neural_cf_model, name=NEURAL_CF)
    print("Model saved successfully!\n")

    # For inferention speedup dataframes and tensors can be precalculated
    print("Precalculating dataframes for users and services...")
    calculate_dfs_for_users_and_services()
    print("Dataframes precalculated successfully!")

    print("Precalculating tensors for users and services...")
    calculate_tensors_for_users_and_services()
    print("Tensors precalculated successfully!")
