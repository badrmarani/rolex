from pathlib import Path

from rolex import cli, init_args
from rolex.baselines import Decoder, Encoder
from rolex.datamodules.janus import JanusDataModule, load_janus
from rolex.models import TabularVAE


# @seed_everything(42)
def main():
    args = init_args(TabularVAE, JanusDataModule)

    exp_name = "janus_bench_dropout"
    version = f"dropout_{args.dropout}"

    # datamodule
    data, _, _, _ = load_janus(args.root)
    dm = JanusDataModule(data=data, **vars(args))
    print(dm.data_dim)
    print(dm.real_filtered_data.shape)

    # encoder/decoder
    args.data_dim = dm.data_dim
    encoder = Encoder(**vars(args))
    decoder = Decoder(**vars(args), mode=args.bayesian_decoder)
    print(encoder)
    print(decoder)

    # model
    model = TabularVAE(
        encoder=encoder,
        decoder=decoder,
        data_transformer=dm.msn,
        real_filtered_data=dm.real_filtered_data,
        **vars(args),
    )

    cli(model, dm, "./", exp_name, args, version)


if __name__ == "__main__":
    main()
