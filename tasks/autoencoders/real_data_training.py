import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from definitions import LOG_DIR
from recommender.engine.datasets.autoencoders import (
    get_autoencoder_dataset_name,
    AUTOENCODERS,
    create_autoencoder_datasets,
)
from recommender.engine.models.autoencoders import (
    create_autoencoder_model,
    USERS_AUTOENCODER,
    SERVICES_AUTOENCODER,
    precalc_embedded_tensors,
)
from recommender.engine.preprocessing import (
    USERS,
    SERVICES,
    precalc_users_and_service_tensors,
)
from recommender.engine.training.autoencoders import (
    train_autoencoder,
    autoencoder_loss_function,
    evaluate_autoencoder,
)
from recommender.engine.utils import (
    load_last_dataset,
    TRAIN,
    VALID,
    TEST,
    save_module,
    save_dataset,
)
from recommender.models import User, Service

from dotenv import load_dotenv

from recommender.services.synthetic_dataset.users import synthesize_users

load_dotenv()
from mongoengine import connect, disconnect

from settings import DevelopmentConfig, get_device

if __name__ == "__main__":
    disconnect()
    connect(host=DevelopmentConfig.MONGODB_HOST)
    device_name = get_device("TRAINING_DEVICE")

    NO_OF_USERS = 2000

    User.objects().delete()
    synthesize_users(NO_OF_USERS)

    precalc_users_and_service_tensors()
    device = torch.device(device_name)

    all_datasets = {AUTOENCODERS: {}}

    for collection_name in (USERS, SERVICES):
        print(f"Creating {collection_name} autoencoder datasets...")
        datasets = create_autoencoder_datasets(
            collection_name,
            train_ds_size=1.0,
            valid_ds_size=0.0,
            device=device
        )
        print(f"{collection_name} autoencoder datasets created successfully!")

        all_datasets[AUTOENCODERS][collection_name] = datasets

        print(f"Saving {collection_name} autoencoder datasets...")
        for split, dataset in datasets.items():
            save_dataset(
                dataset, name=f"{AUTOENCODERS} {collection_name} {split} dataset"
            )
        print(f"{collection_name} autoencoder datasets saved successfully!")

    device = "cpu"
    writer = SummaryWriter(log_dir=LOG_DIR)
    # writer = None

    UOH = len(User.objects[0].tensor)  # synthetic=True
    SOH = len(Service.objects.first().tensor)

    UE = 32
    SE = 64

    # USERS AUTOENCODER
    user_autoencoder_train_ds = load_last_dataset(
        get_autoencoder_dataset_name(USERS, TRAIN)
    )
    # user_autoencoder_valid_ds = load_last_dataset(
    #     get_autoencoder_dataset_name(USERS, VALID)
    # )
    # user_autoencoder_test_ds = load_last_dataset(
    #     get_autoencoder_dataset_name(USERS, TEST)
    # )

    USER_AE_BATCH_SIZE = 128

    user_autoencoder_train_ds_dl = DataLoader(
        user_autoencoder_train_ds, batch_size=USER_AE_BATCH_SIZE, shuffle=True
    )
    # user_autoencoder_valid_ds_dl = DataLoader(
    #     user_autoencoder_valid_ds, batch_size=USER_AE_BATCH_SIZE, shuffle=True
    # )
    # user_autoencoder_test_ds_dl = DataLoader(
    #     user_autoencoder_test_ds, batch_size=USER_AE_BATCH_SIZE, shuffle=True
    # )

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

    EPOCHS = 200

    trained_user_autoencoder_model = train_autoencoder(
        model=user_autoencoder_model,
        optimizer=optimizer,
        loss_function=autoencoder_loss_function,
        epochs=EPOCHS,
        train_ds_dl=user_autoencoder_train_ds_dl,
        # valid_ds_dl=user_autoencoder_valid_ds_dl,
        writer=writer,
        save_period=10,
        verbose=True,
        device=device,
    )

    loss = evaluate_autoencoder(
        trained_user_autoencoder_model,
        # user_autoencoder_test_ds_dl,
        user_autoencoder_train_ds_dl,
        autoencoder_loss_function,
        device,
    )
    print(f"User Autoencoder testing loss: {loss}")

    save_module(trained_user_autoencoder_model, name=USERS_AUTOENCODER)



    # SERVICE AUTOENCODER
    service_autoencoder_train_ds = load_last_dataset(
        get_autoencoder_dataset_name(SERVICES, TRAIN)
    )
    # service_autoencoder_valid_ds = load_last_dataset(
    #     get_autoencoder_dataset_name(SERVICES, VALID)
    # )
    # service_autoencoder_test_ds = load_last_dataset(
    #     get_autoencoder_dataset_name(SERVICES, TEST)
    # )

    SERVICE_AE_BATCH_SIZE = 128

    service_autoencoder_train_ds_dl = DataLoader(
        service_autoencoder_train_ds, batch_size=SERVICE_AE_BATCH_SIZE, shuffle=True
    )
    # service_autoencoder_valid_ds_dl = DataLoader(
    #     service_autoencoder_valid_ds, batch_size=SERVICE_AE_BATCH_SIZE, shuffle=True
    # )
    # service_autoencoder_test_ds_dl = DataLoader(
    #     service_autoencoder_test_ds, batch_size=SERVICE_AE_BATCH_SIZE, shuffle=True
    # )

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
        # valid_ds_dl=service_autoencoder_valid_ds_dl,
        writer=writer,
        save_period=10,
        verbose=True,
        device=device,
    )

    loss = evaluate_autoencoder(
        trained_service_autoencoder_model,
        # service_autoencoder_test_ds_dl,
        service_autoencoder_train_ds_dl,
        autoencoder_loss_function,
        device,
    )
    print(f"Service Autoencoder testing loss: {loss}")

    save_module(trained_service_autoencoder_model, name=SERVICES_AUTOENCODER)

    precalc_embedded_tensors(USERS)
    precalc_embedded_tensors(SERVICES)
