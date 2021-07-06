# pylint: disable=redefined-builtin, no-member, redefined-outer-name,
# pylint: disable=too-many-arguments, too-many-locals, invalid-name
# pylint: disable=unused-variable, too-many-statements
# pylint: disable=fixme

"""Implementation of the RL Agent training"""

# TODO: models reloading callback

from torch.optim import Adam
from torch.utils.data import DataLoader

from recommender.engine.agents.rl_agent.models.actor import Actor, ACTOR_V1, ACTOR_V2
from recommender.engine.agents.rl_agent.models.history_embedder import (
    HistoryEmbedder,
    HISTORY_EMBEDDER_V1,
    HISTORY_EMBEDDER_V2,
)

from recommender.engine.datasets.autoencoders import (
    get_autoencoder_dataset_name,
)
from recommender.engine.models.autoencoders import (
    create_autoencoder_model,
    SERVICES_AUTOENCODER,
    USERS_AUTOENCODER,
)
from recommender.engine.training.autoencoders import (
    train_autoencoder,
    autoencoder_loss_function,
    evaluate_autoencoder,
)
from recommender.models import Service, User
from recommender.engine.utils import save_module, load_last_dataset, TRAIN, VALID, TEST
from recommender.engine.preprocessing import USERS, SERVICES
from settings import get_device


def rl_agent_training() -> None:
    """
    This function proceed RL Agent training.

    It loads needed data from the database and perform training of the rl agent.
    In the end it save actor_model, history_embedder and critic_model to the database.
    """

    device = get_device("TRAINING_DEVICE")
    writer = None  # TODO use SummaryWriter when docker-tensorboard issue is solved

    # USERS AUTOENCODER
    user_autoencoder_train_ds = load_last_dataset(
        get_autoencoder_dataset_name(USERS, TRAIN)
    )
    user_autoencoder_valid_ds = load_last_dataset(
        get_autoencoder_dataset_name(USERS, VALID)
    )
    user_autoencoder_test_ds = load_last_dataset(
        get_autoencoder_dataset_name(USERS, TEST)
    )

    USER_AE_BATCH_SIZE = 128

    user_autoencoder_train_ds_dl = DataLoader(
        user_autoencoder_train_ds, batch_size=USER_AE_BATCH_SIZE, shuffle=True
    )
    user_autoencoder_valid_ds_dl = DataLoader(
        user_autoencoder_valid_ds, batch_size=USER_AE_BATCH_SIZE, shuffle=True
    )
    user_autoencoder_test_ds_dl = DataLoader(
        user_autoencoder_test_ds, batch_size=USER_AE_BATCH_SIZE, shuffle=True
    )

    USER_FEATURES_DIM = len(User.objects[0].tensor)
    USER_EMBEDDING_DIM = 32

    user_autoencoder_model = create_autoencoder_model(
        USERS,
        features_dim=USER_FEATURES_DIM,
        embedding_dim=USER_EMBEDDING_DIM,
        writer=writer,
        train_ds_dl=user_autoencoder_train_ds_dl,
        device=device,
    )

    LR = 0.01
    optimizer = Adam(user_autoencoder_model.parameters(), lr=LR)

    EPOCHS = 2000

    trained_user_autoencoder_model = train_autoencoder(
        model=user_autoencoder_model,
        optimizer=optimizer,
        loss_function=autoencoder_loss_function,
        epochs=EPOCHS,
        train_ds_dl=user_autoencoder_train_ds_dl,
        valid_ds_dl=user_autoencoder_valid_ds_dl,
        writer=writer,
        save_period=10,
        verbose=True,
        device=device,
    )

    loss = evaluate_autoencoder(
        trained_user_autoencoder_model,
        user_autoencoder_test_ds_dl,
        autoencoder_loss_function,
        device,
    )
    print(f"User Autoencoder testing loss: {loss}")

    save_module(trained_user_autoencoder_model, name=USERS_AUTOENCODER)

    # SERVICE AUTOENCODER
    service_autoencoder_train_ds = load_last_dataset(
        get_autoencoder_dataset_name(SERVICES, TRAIN)
    )
    service_autoencoder_valid_ds = load_last_dataset(
        get_autoencoder_dataset_name(SERVICES, VALID)
    )
    service_autoencoder_test_ds = load_last_dataset(
        get_autoencoder_dataset_name(SERVICES, TEST)
    )

    SERVICE_AE_BATCH_SIZE = 128

    service_autoencoder_train_ds_dl = DataLoader(
        service_autoencoder_train_ds, batch_size=SERVICE_AE_BATCH_SIZE, shuffle=True
    )
    service_autoencoder_valid_ds_dl = DataLoader(
        service_autoencoder_valid_ds, batch_size=SERVICE_AE_BATCH_SIZE, shuffle=True
    )
    service_autoencoder_test_ds_dl = DataLoader(
        service_autoencoder_test_ds, batch_size=SERVICE_AE_BATCH_SIZE, shuffle=True
    )

    SERVICE_FEATURES_DIM = len(Service.objects[0].tensor)
    SERVICE_EMBEDDING_DIM = 64

    service_autoencoder_model = create_autoencoder_model(
        SERVICES,
        features_dim=SERVICE_FEATURES_DIM,
        embedding_dim=SERVICE_EMBEDDING_DIM,
        writer=writer,
        train_ds_dl=service_autoencoder_train_ds_dl,
        device=device,
    )

    LR = 0.01
    optimizer = Adam(service_autoencoder_model.parameters(), lr=LR)

    EPOCHS = 2000

    trained_service_autoencoder_model = train_autoencoder(
        model=service_autoencoder_model,
        optimizer=optimizer,
        loss_function=autoencoder_loss_function,
        epochs=EPOCHS,
        train_ds_dl=service_autoencoder_train_ds_dl,
        valid_ds_dl=service_autoencoder_valid_ds_dl,
        writer=writer,
        save_period=10,
        verbose=True,
        device=device,
    )

    loss = evaluate_autoencoder(
        trained_service_autoencoder_model,
        service_autoencoder_test_ds_dl,
        autoencoder_loss_function,
        device,
    )
    print(f"Service Autoencoder testing loss: {loss}")

    save_module(trained_service_autoencoder_model, name=SERVICES_AUTOENCODER)

    # Actor Critic training
    UE = USER_EMBEDDING_DIM

    SE = SERVICE_EMBEDDING_DIM

    actor_v1_history_embedder = HistoryEmbedder(SE=SE, num_layers=3, dropout=0.5)

    actor_v1 = Actor(
        K=3,
        SE=SE,
        UE=UE,
        I=len(Service.objects),
        history_embedder=actor_v1_history_embedder,
    )
    # TODO: actor_v1 (and critic) training

    actor_v2_history_embedder = HistoryEmbedder(SE=SE, num_layers=3, dropout=0.5)

    actor_v2 = Actor(
        K=2,
        SE=SE,
        UE=UE,
        I=len(Service.objects),
        history_embedder=actor_v2_history_embedder,
    )
    # TODO: actor_v2 (and critic) training

    save_module(module=actor_v1_history_embedder, name=HISTORY_EMBEDDER_V1)
    save_module(module=actor_v1, name=ACTOR_V1)
    save_module(module=actor_v2_history_embedder, name=HISTORY_EMBEDDER_V2)
    save_module(module=actor_v2, name=ACTOR_V2)
