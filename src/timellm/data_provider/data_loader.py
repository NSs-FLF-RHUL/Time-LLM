# ruff: noqa: N801

"""Datasets in timellm's format."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from timellm.data_provider.m4 import M4Dataset, M4Meta
from timellm.utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    """ETT hourly dataset."""

    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        size: list | np.ndarray | None = None,
        features: str = "S",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        timeenc: int = 0,
        freq: str = "h",
        percent: float = 100,
        *,
        scale: bool = True,
    ) -> None:
        """Initialize EETh dataset."""
        if flag not in ["train", "test", "val"]:
            invalid_flag_err_msg = f"Invalid flag: {flag}."
            raise ValueError(invalid_flag_err_msg)
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.percent = percent
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
        """Read ETTh data."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(Path(self.root_path) / self.data_path)

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

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple:
        """
        Get ETTh dataset item.

        :param index: item index.
        """
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
        """Compute ETTH dataset length."""
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Denormalize data.

        :param data: ETTh data to denormalize.
        """
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    """ETT minutly dataset."""

    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        size: list | np.ndarray | None = None,
        features: str = "S",
        data_path: str = "ETTm1.csv",
        target: str = "OT",
        timeenc: int = 0,
        freq: str = "t",
        percent: float = 100,
        *,
        scale: bool = True,
    ) -> None:
        """Initialize ETTm dataset."""
        if flag not in ["train", "test", "val"]:
            invalid_flag_err_msg = f"Invalid flag: {flag}."
            raise ValueError(invalid_flag_err_msg)
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.percent = percent
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
        """Read ETTm data."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(Path(self.root_path) / self.data_path)

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

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
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
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple:
        """
        Get ETTm dataset item.

        :param index: item index.
        """
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
        """Compute ETTm dataset length."""
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Denormalize data.

        :param data: ETTm data to denormalize.
        """
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    """Custom dataset."""

    def __init__(
        self,
        root_path: str | Path,
        flag: str = "train",
        size: list | np.ndarray | None = None,
        features: str = "S",
        data_path: str | Path = "ETTh1.csv",
        target: str = "OT",
        timeenc: int = 0,
        freq: str = "h",
        percent: float = 100,
        seasonal_patterns: str|None = None,
        nrows: int | None = None,
        *,
        scale: bool = True,
    ) -> None:
        """Initialize custom dataset."""
        if flag not in ["train", "test", "val"]:
            invalid_flag_err_msg = f"Invalid flag: {flag}."
            raise ValueError(invalid_flag_err_msg)
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.nrows = nrows
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self) -> None:
        """Read custom dataset."""
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(Path(self.root_path) / self.data_path, nrows=self.nrows)

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[['date'] + cols + [self.target]] #df_raw["date", *cols, self.target]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = int((border2 - self.seq_len) * self.percent // 100 + self.seq_len)

        if self.features in ["M", "MS"]:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index: int) -> tuple:
        """Get custom dataset item."""
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
        """Conpute custom dataset length."""
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Denormalize data.

        :param: custom data to denormalize
        """
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    """M4 dataset."""

    def __init__(
        self,
        root_path: str | Path,
        size: list | np.ndarray,
        flag: str = "pred",
        features: str = "S",
        target: str = "OT",
        timeenc: int = 0,
        seasonal_patterns: str = "Yearly",
        *,
        scale: bool = False,
        inverse: bool = False,
    ) -> None:
        """Initialize M4 dataset."""
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self) -> None:
        """Read M4 data."""
        if self.flag == "train":
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [
                v[~np.isnan(v)]
                for v in dataset.vals[dataset.groups == self.seasonal_patterns]
            ]
        )  # split different frequencies
        self.ids = np.array(dataset.ids[dataset.groups == self.seasonal_patterns])
        self.timeseries = list(training_values)

    def __getitem__(self, index: int) -> tuple:
        """
        Get M4 dataset item.

        :param index: item index.
        """
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.default_rng().integers(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
            high=len(sampled_timeseries),
            size=1,
        )[0]

        insample_window = sampled_timeseries[
            max(0, cut_point - self.seq_len) : cut_point
        ]
        insample[-len(insample_window) :, 0] = insample_window
        insample_mask[-len(insample_window) :, 0] = 1.0
        outsample_window = sampled_timeseries[
            cut_point - self.label_len : min(
                len(sampled_timeseries), cut_point + self.pred_len
            )
        ]
        outsample[: len(outsample_window), 0] = outsample_window
        outsample_mask[: len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self) -> int:
        """Compute M4 dataset length."""
        return len(self.timeseries)

    def last_insample_window(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return last window of insample size of all timeseries.

        This function does not support batching and does not
        reshuffle timeseries.

        :return: Last insample window of all timeseries.
        Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len :]
            insample[i, -len(ts) :] = ts_last_window
            insample_mask[i, -len(ts) :] = 1.0
        return insample, insample_mask
