from rolex import init_args
from rolex.datamodules.janus import JanusDataModule, load_janus
from rolex.models import TabularVAE


def main():
    args = init_args(None, JanusDataModule)
    data, _, _, _ = load_janus(args.root)
    JanusDataModule(data=data, **vars(args))


if __name__ == "__main__":
    main()
