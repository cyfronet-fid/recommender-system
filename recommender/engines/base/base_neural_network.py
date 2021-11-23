# pylint: disable=too-few-public-methods, line-too-long, too-many-arguments
"""Abstract neural network"""

from abc import ABC
from typing import Tuple
from itertools import chain

from torch.nn import Linear, BatchNorm1d, ReLU


class BaseNeuralNetwork(ABC):
    """Abstract Neural Network"""

    def _create_layers(
        self,
        input_dim: int,
        output_dim: int,
        layers_sizes: Tuple[int],
        inc_batchnorm: bool = True,
        activation=ReLU,
    ):
        """
        Create neural network layers based on a provided tuple which contains the size of layers

        Args:
            input_dim: input dimensionality,
            output_dim: output dimensionality,
            layers_sizes: a tuple of target layers sizes,
            inc_batchnorm: include BatchNorm1d?
            activation: activation function
        """
        # The first layer
        layers = self._create_layer(
            input_dim=input_dim,
            output_dim=layers_sizes[0],
            inc_batchnorm=inc_batchnorm,
            activation=activation,
        )

        # The middle layers
        layers += list(
            chain.from_iterable(
                [
                    self._create_layer(
                        input_dim=n_size,
                        output_dim=next_n_size,
                        inc_batchnorm=inc_batchnorm,
                        activation=activation,
                    )
                    for n_size, next_n_size in zip(layers_sizes, layers_sizes[1:])
                ]
            )
        )

        if output_dim:
            # The last layer
            layers += [Linear(layers_sizes[-1], output_dim)]

        return layers

    @staticmethod
    def _create_layer(input_dim: int, output_dim: int, inc_batchnorm: bool, activation):
        """
        Create a single neural network layer

        Args:
            input_dim: input dimensionality,
            output_dim: output dimensionality,
            inc_batchnorm: include BatchNorm1d?
            activation: activation function
        """
        layer = [
            Linear(input_dim, output_dim),
            BatchNorm1d(output_dim),
            activation(),
        ]
        if not inc_batchnorm:
            del layer[1]

        return layer
