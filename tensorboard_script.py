import os

from definitions import LOG_DIR

if __name__ == "__main__":
    os.system(f"tensorboard --logdir={LOG_DIR}")
