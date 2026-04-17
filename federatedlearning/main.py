import argparse

from federatedlearning.server.kd_dkd_server import run_federated_kd_dkd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "ten"],
        default="all",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["KD", "DKD"],
        default="KD",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--global-rounds",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--kd-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dkd-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--tckd-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--nckd-weight",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./fed_history",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="niid-kd-dkd",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_federated_kd_dkd(args)


if __name__ == "__main__":
    main()
