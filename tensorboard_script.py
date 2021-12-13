# pylint: disable=missing-module-docstring, fixme

import os

from definitions import RUN_DIR

# TODO: Implement as a flask task
if __name__ == "__main__":
    os.system(f"tensorboard --logdir={RUN_DIR}")
