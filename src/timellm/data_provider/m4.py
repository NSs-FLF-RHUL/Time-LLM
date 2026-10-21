# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""M4 Dataset."""

import logging
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib import request

import numpy as np
import pandas as pd


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split("/")[-1] if len(url) > 0 else ""


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count: float, block_size: float, total_size: float) -> None:
        """Print download progress."""
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(f"\rDownloading {url} to {file_path} {progress_pct:.1f}%")
        sys.stdout.flush()

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not pathlib.Path(file_path).is_file():
        opener = request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        request.install_opener(opener)
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write("\n")
        sys.stdout.flush()
        file_info = pathlib.Path(f).stat()
        logger.info(
            "Successfully downloaded %s %f bytes.",
            pathlib.Path(file_path).name,
            file_info.st_size,
        )
    else:
        file_info = pathlib.Path(file_path).stat()
        logger.info("File already exists: %s %f bytes.", file_path, file_info.st_size)


@dataclass()
class M4Dataset:
    """M4 Dataset."""

    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    vals: np.ndarray

    @staticmethod
    def load(
        dataset_file: str | Path = "../dataset/m4", *, training: bool = True
    ) -> "M4Dataset":
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        info_file = pathlib.Path(dataset_file) / "M4-info.csv"
        train_cache_file = pathlib.Path(dataset_file) / "training.npz"
        test_cache_file = pathlib.Path(dataset_file) / "test.npz"
        m4_info = pd.read_csv(info_file)
        return M4Dataset(
            ids=m4_info.M4id.values,
            groups=m4_info.SP.values,
            frequencies=m4_info.Frequency.values,
            horizons=m4_info.Horizon.values,
            vals=np.load(
                train_cache_file if training else test_cache_file, allow_pickle=True
            ),
        )


@dataclass()
class M4Meta:
    """M4 Meta."""

    def __init__(self) -> None:
        """Initialize M4 Meta."""
        self.seasonal_patterns = [
            "Yearly",
            "Quarterly",
            "Monthly",
            "Weekly",
            "Daily",
            "Hourly",
        ]
        self.horizons = [6, 8, 18, 13, 14, 48]
        self.frequencies = [1, 4, 12, 1, 1, 24]
        self.horizons_map = {
            "Yearly": 6,
            "Quarterly": 8,
            "Monthly": 18,
            "Weekly": 13,
            "Daily": 14,
            "Hourly": 48,
        }  # different predict length
        self.frequency_map = {
            "Yearly": 1,
            "Quarterly": 4,
            "Monthly": 12,
            "Weekly": 1,
            "Daily": 1,
            "Hourly": 24,
        }
        self.history_size = {
            "Yearly": 1.5,
            "Quarterly": 1.5,
            "Monthly": 1.5,
            "Weekly": 10,
            "Daily": 10,
            "Hourly": 10,
        }  # from interpretable.gin
