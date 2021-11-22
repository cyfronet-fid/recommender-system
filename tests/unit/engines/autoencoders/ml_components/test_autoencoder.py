# pylint: disable-all
from random import randint
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

from recommender.engines.autoencoders.ml_components.autoencoder import AutoEncoder


def test_autoencoder():
    """
    Testing:
    -> Creating encoder/decoder models using a tuple of layer sizes
    """
    features_dim = randint(2, 100)
    embedding_dim = randint(2, 100)

    encoder_layer_sizes = (128, 64)
    decoder_layer_sizes = (64, 128)

    # Create encoder and decoder using tuple of integers
    models = AutoEncoder(
        features_dim, embedding_dim, encoder_layer_sizes, decoder_layer_sizes
    )

    encoder = models.encoder
    decoder = models.decoder

    # Create "handmade" encoder and decoder
    handmade_encoder = Sequential(
        Linear(features_dim, 128),
        BatchNorm1d(128),
        ReLU(),
        Linear(128, 64),
        BatchNorm1d(64),
        ReLU(),
        Linear(64, embedding_dim),
    )

    handmade_decoder = Sequential(
        Linear(embedding_dim, 64),
        BatchNorm1d(64),
        ReLU(),
        Linear(64, 128),
        BatchNorm1d(128),
        ReLU(),
        Linear(128, features_dim),
    )

    # Those encoders and decoders should be the same
    assert str(encoder) == str(handmade_encoder)
    assert str(decoder) == str(handmade_decoder)
