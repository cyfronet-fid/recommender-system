# pylint: disable-all
# from recommender.engine.datasets.autoencoders import create_autoencoder_datasets
#
# device = get_device("TRAINING_DEVICE")
# writer = None  # TODO use SummaryWriter when docker-tensorboard issue is solved
#
# # USERS AUTOENCODER
#
# # Data Extraction
# users = User.object(synthetic=True/False, ...)
#
#
# # Data Validation
# assert len(users) >= 100 # magic constant XD
# # czy oni maja jakiekolwiek kategorie/scientific domains
# # maybe more
#
#
# # Data Preparation
# users_one_hot = precalc(users)
#
#
# # scamwchuj one hoty
# USER_FEATURES_DIM = len(users[0].one_hot_tensor)
#
# ds_tensor = torch.Tensor(users_one_hot)
# ds_tensor = ds_tensor.to(device)
# dataset = torch.utils.data.TensorDataset(ds_tensor)
#
# USER_AE_BATCH_SIZE = 128
#
# user_autoencoder_train_ds_dl = DataLoader(
#     dataset, batch_size=USER_AE_BATCH_SIZE, shuffle=True
# )
#
# # Model training
# USER_EMBEDDING_DIM = 32
#
# user_autoencoder_model = create_autoencoder_model(
#     USERS,
#     features_dim=USER_FEATURES_DIM,
#     embedding_dim=USER_EMBEDDING_DIM,
#     writer=writer,
#     train_ds_dl=user_autoencoder_train_ds_dl,
#     device=device,
# )
#
# LR = 0.01
# optimizer = Adam(user_autoencoder_model.parameters(), lr=LR)
#
# EPOCHS = 2000
#
# trained_user_autoencoder_model = train_autoencoder(
#     model=user_autoencoder_model,
#     optimizer=optimizer,
#     loss_function=autoencoder_loss_function,
#     epochs=EPOCHS,
#     train_ds_dl=user_autoencoder_train_ds_dl,
#     valid_ds_dl=user_autoencoder_valid_ds_dl,
#     writer=writer,
#     save_period=10,
#     verbose=True,
#     device=device,
# )
#
# # Model evaluation
# loss = evaluate_autoencoder(
#     trained_user_autoencoder_model,
#     user_autoencoder_test_ds_dl,
#     autoencoder_loss_function,
#     device,
# )
# print(f"User Autoencoder testing loss: {loss}") # and other metrics
#
# # Model validation
# # maybe some some speed check, memory usage, etc. it has to better than some baselines
#
#     save_module(trained_user_autoencoder_model, name=USERS_AUTOENCODER)
