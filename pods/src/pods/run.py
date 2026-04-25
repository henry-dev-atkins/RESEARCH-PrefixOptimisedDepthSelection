from __future__ import annotations

import argparse

from pods.config.schema import load_config
from pods.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
