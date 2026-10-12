"""Train the hierarchical multi-player video model."""

from multics.training_args import parse_args


def main():
    args = parse_args()
    import lightning as pl

    pl.seed_everything(args.seed, workers=True)
    from multics.training import train

    train(args)


if __name__ == "__main__":
    main()
