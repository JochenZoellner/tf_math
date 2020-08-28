import argparse
import logging

logger = logging.getLogger(__name__)

MODULE_NAME = "python_template"


def main(args):
    logger.info(f"Running main() of {MODULE_NAME}")
    pass


def parse_args(args=None):
    parser = argparse.ArgumentParser(f"Parser of '{MODULE_NAME}'")
    parser.add_argument("directory", type=str, default="models", help="directory to search recursively for checkpoints")
    args_ = parser.parse_args(args)
    return args_


if __name__ == "__main__":
    logger.setLevel("INFO")
    logger.info(f"Running {MODULE_NAME} as __main__...")
    arguments = parse_args()
    main(args=arguments)
