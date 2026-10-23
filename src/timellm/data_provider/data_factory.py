"""Prepare datasets for neural networks."""

from argparse import Namespace

import torch

from timellm.data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_M4,
)
from timellm.utils.tools import seed_worker

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "ECL": Dataset_Custom,
    "Traffic": Dataset_Custom,
    "Weather": Dataset_Custom,
    "m4": Dataset_M4,
    "Pulsar": Dataset_Custom,
}


def data_provider(
    args: Namespace,
    flag: "str",
    seed: int | None = None,
    data_set: torch.utils.data.Dataset | None = None,
    *,
    scale: bool = True,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.DataLoader]:
    """
    Create (dataset and) dataloader.

    If data_set is not provided, data are read from the path indicated in args. This
    function can be called several times with data_set = None to load data from
    different files. Several datasets can then be merged and passed as argument
    to this function to create a unique dataloader for an extended dataset.

    :param args: run configuration.
    :param flag: data type (train, test, etc.).
    :param seed: seed for random number generator.
    :param data_set: data set, optional.
    :param scale: whether to scale the data (default: True).
    :return: PyTorch dataset and dataloader.
    """
    data = data_dict[args.data]
    timeenc = 0 if args.embed != "timeF" else 1
    percent = args.percent

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if data_set is None:
        if args.data == "m4":
            drop_last = False
            data_set = data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                seasonal_patterns=args.seasonal_patterns,
            )
        else:
            data_set = data(
                root_path=args.root_path,
                data_path=args.data_path,
                flag=flag,
                size=[args.seq_len, args.label_len, args.pred_len],
                features=args.features,
                target=args.target,
                timeenc=timeenc,
                freq=freq,
                percent=percent,
                seasonal_patterns=args.seasonal_patterns,
                skiprows=args.skiprows,
                nrows=args.nrows,
                scale=scale,
            )

    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
        wif = seed_worker
    else:
        g = None
        wif = None
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=wif,
        generator=g,
    )
    return data_set, data_loader
