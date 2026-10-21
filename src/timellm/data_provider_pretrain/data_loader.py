# ruff: noqa: N801, D102, ARG002, FBT001, FBT002, D100
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from timellm.utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    """Hourly ETT dataset."""

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: list | None = None,
        features: str = "S",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "h",
        percent: float = 100,
        seasonal_patterns: int | None = None,
        pretrain: bool = True,
    ) -> None:
        """Initialize dataset."""
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        if flag not in ["train", "test", "val"]:
            err_msg = f"flag should be train, test, not {flag}."
            raise ValueError(err_msg)
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.pretrain = pretrain
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self) -> None:
        """Read data from csv file."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(Path(self.root_path) / Path(self.data_path))

        if self.pretrain:
            border1s = [
                0,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        else:
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = int((border2 - self.seq_len) * self.percent // 100 + self.seq_len)

        if self.features in ["M", "MS"]:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            err_msg = f"features should be M, MS or S, not {self.features}."
            raise ValueError(err_msg)

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.to_numpy())
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).to_numpy()
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)
        else:
            err_msg = f"timeenc should be 0 or 1, not {self.timeenc}."
            raise ValueError(err_msg)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple:
        """Get data at given index."""
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        """Compute length of dataset."""
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    """Dataset ETT minute."""

    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: list | np.ndarray | None = None,
        features: str = "S",
        data_path: str = "ETTm1.csv",
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 0,
        freq: str = "t",
        percent: float = 100,
        seasonal_patterns: None = None,
        pretrain: bool = True,
    ) -> None:
        """Initialize dataset."""
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        if flag not in ["train", "test", "val"]:
            err_msg = f"flag should be train, test, or val, no {flag}."
            raise ValueError(err_msg)
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.pretrain = pretrain
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self) -> None:
        """Read data."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(Path(self.root_path) / Path(self.data_path))

        if self.pretrain:
            border1s = [
                0,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:
            border1s = [
                0,
                12 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = int((border2 - self.seq_len) * self.percent // 100 + self.seq_len)

        if self.features in ["M", "MS"]:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            err_msg = f"features should be M, MS or S, not {self.features}."
            raise ValueError(err_msg)

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.to_numpy())
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).to_numpy()
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].to_numpy()), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        else:
            err_msg = f"timeenc should be 0 or 1, not {self.timeenc}."
            raise ValueError(err_msg)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple:
        """Get dataset item."""
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        """Compute dataset length."""
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        return self.scaler.inverse_transform(data)
