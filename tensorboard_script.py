# pylint: disable=missing-module-docstring, fixme

import os

from definitions import LOG_DIR

# TODO: Implement as a flask task
if __name__ == "__main__":
    os.system(f"tensorboard --logdir={LOG_DIR}")
