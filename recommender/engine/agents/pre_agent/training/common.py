# pylint: disable=redefined-builtin, no-member, redefined-outer-name,
# pylint: disable=too-many-arguments, too-many-locals, invalid-name
# pylint: disable=unused-variable, too-many-statements
# pylint: disable=fixme, line-too-long

"""Training and evaluation related functions"""

import torch
from sklearn.metrics import classification_report
from torch.optim import Adam
from torch.utils.data import DataLoader

from recommender.engine.agents.pre_agent.datasets.neural_collaborative_filtering import (
    get_ncf_dataset_name,
)
from recommender.engine.datasets.autoencoders import (
    get_autoencoder_dataset_name,
)
from recommender.engine.models.autoencoders import (
    create_autoencoder_model,
    create_embedder,
    SERVICES_AUTOENCODER,
    USERS_AUTOENCODER,
)

from recommender.engine.agents.pre_agent.models.neural_colaborative_filtering import (
    create_nfc_model,
    get_preds_for_ds,
    NEURAL_CF,
)
from recommender.engine.training.autoencoders import (
    train_autoencoder,
    autoencoder_loss_function,
    evaluate_autoencoder,
)
from recommender.engine.agents.pre_agent.training.neural_collaborative_filtering import (
    train_ncf,
    evaluate_ncf,
    accuracy_function,
)
from recommender.models import Service, User
from recommender.engine.utils import save_module, load_last_dataset, TRAIN, VALID, TEST
from recommender.engine.preprocessing import USERS, SERVICES
from settings import get_device


def pre_agent_training():
    """Main full training function. It load datasets, creates model, loss
    function and optimizer. It trains model, show train/valid accuracy/loss
    and evaluate model on the test set. In the end it saves model.

    This function is used in all tasks and is also called in the training endpoint.
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

    # NEURAL COLLABORATIVE FILTERING
    ncf_train_ds = load_last_dataset(get_ncf_dataset_name(TRAIN))
    ncf_valid_ds = load_last_dataset(get_ncf_dataset_name(VALID))
    ncf_test_ds = load_last_dataset(get_ncf_dataset_name(TEST))

    BATCH_SIZE = 32

    ncf_train_ds_dl = DataLoader(ncf_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    ncf_valid_ds_dl = DataLoader(ncf_valid_ds, batch_size=BATCH_SIZE, shuffle=True)
    ncf_test_ds_dl = DataLoader(ncf_test_ds, batch_size=BATCH_SIZE, shuffle=True)

    service_embedder = create_embedder(trained_service_autoencoder_model)
    user_embedder = create_embedder(trained_user_autoencoder_model)

    USERS_MAX_ID = User.objects.order_by("-id").first().id
    SERVICES_MAX_ID = Service.objects.order_by("-id").first().id

    MF_EMBEDDING_DIM = 64
    USER_IDS_EMBEDDING_DIM = 64
    SERVICE_IDS_EMBEDDING_DIM = 64

    MLP_LAYERS_SPEC = [64, 32, 16, 8]
    CONTENT_MLP_LAYERS_SPEC = [128, 64, 32]

    ncf_model = create_nfc_model(
        users_max_id=USERS_MAX_ID,
        services_max_id=SERVICES_MAX_ID,
        mf_embedding_dim=MF_EMBEDDING_DIM,
        user_ids_embedding_dim=USER_IDS_EMBEDDING_DIM,
        service_ids_embedding_dim=SERVICE_IDS_EMBEDDING_DIM,
        user_embedder=user_embedder,
        service_embedder=service_embedder,
        mlp_layers_spec=MLP_LAYERS_SPEC,
        content_mlp_layers_spec=CONTENT_MLP_LAYERS_SPEC,
        writer=writer,
        train_ds_dl=ncf_train_ds_dl,
        device=device,
    )

    ncf_loss_function = torch.nn.BCELoss()

    LR = 0.01
    ncf_optimizer = Adam(ncf_model.parameters(), lr=LR)

    EPOCHS = 500

    trained_ncf_model = train_ncf(
        model=ncf_model,
        optimizer=ncf_optimizer,
        loss_function=ncf_loss_function,
        epochs=EPOCHS,
        train_ds_dl=ncf_train_ds_dl,
        valid_ds_dl=ncf_valid_ds_dl,
        save_period=10,
        writer=writer,
        verbose=True,
        device=device,
    )

    loss, acc = evaluate_ncf(
        trained_ncf_model, ncf_test_ds_dl, ncf_loss_function, accuracy_function, device
    )
    print(f"Testing loss: {loss}, testing accuracy: {acc}")

    labels, preds = get_preds_for_ds(trained_ncf_model, ncf_test_ds)
    print(classification_report(labels, preds > 0.5))

    save_module(trained_ncf_model, name=NEURAL_CF)

    # TODO: models reloading callback
