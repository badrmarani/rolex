from pathlib import Path

from rolex import cli, init_args
from rolex.baselines import Decoder, Encoder
from rolex.datamodules.janus import JanusDataModule, load_janus
from rolex.models import TabularVAE


def main():
    args = init_args(TabularVAE, JanusDataModule)

    exp_name = "test_janus_best"
    version = f"dropout_p{args.dropout}_b{args.beta_on_kld}"

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
        mode=args.bayesian_decoder,
        **vars(args),
    )

    cli(model, dm, "./", exp_name, args, version)


if __name__ == "__main__":
    main()
