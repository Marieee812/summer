import logging
import numpy
import torch
import torchvision.transforms.functional as TF
import urllib.request
import yaml
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Callable
from zipfile import ZipFile

from summer.utils.stat import DatasetStat, compute_stat



class CellTrackingChallengeDataset(Dataset):
    download_link: str

    def __init__(
        self,
        one: bool,
        two: bool,
        labeled_only: bool,
        transform: Callable[[Image.Image, Image.Image, DatasetStat], Tuple[torch.Tensor, torch.Tensor]],
    ):
        if not (one or two):
            raise ValueError("No subdataset selected. Choose 'one' and/or 'two'")

        folder = self.download_link.split("/")[-1].replace(".zip", "/")
        self.logger = logging.getLogger(folder)
        self.path: Path = Path(__file__).parent.parent / "data" / folder
        if not (self.path).exists():
            with urllib.request.urlopen(self.download_link) as response, ZipFile(BytesIO(response.read())) as zipf:
                assert folder in zipf.namelist(), zipf.namelist()
                zipf.extractall(path=self.path.parent)

        # collect valid indices
        self.indices = []
        self.stats = []
        self.folder_names = []
        self._len = 0
        for name, selected in zip(["01", "02"], [one, two]):
            if not selected:
                continue

            self.folder_names.append(name)
            if labeled_only:
                name += "_GT/SEG"

            folder = self.path / name
            self.indices.append([int(file_name.name[-7:-4]) for file_name in folder.glob("*.tif")])
            self._len += len(self.indices[-1])

            stat_path = folder / "stat.yml"
            if stat_path.exists():
                with stat_path.open() as f:
                    stat = DatasetStat(**yaml.safe_load(f))
            else:
                self.transform = lambda img, *args: (TF.to_tensor(img), *args)
                stat = compute_stat(self)
                with stat_path.open("w") as f:
                    yaml.safe_dump(asdict(stat), f)

            self.stats.append(stat)

        self.transform = transform

    def __len__(self):
        return self._len

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_item = item
        for i, idx in enumerate(self.indices):
            if item < len(idx):
                folder = self.folder_names[i]
                stat_exists = self.stats[i : i + 1]
                return self.transform(
                    Image.open(self.path / folder / f"t{idx[item]:03}.tif"),
                    Image.open(self.path / f"{folder}_GT/SEG/man_seg{idx[item]:03}.tif"),
                    stat_exists[0] if stat_exists else None,
                )
            else:
                item -= len(idx)

        raise IndexError(original_item)


class DIC_C2DH_HeLa(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip"


class Fluo_C2DL_MSC(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/Fluo-C2DL-MSC.zip"


class Fluo_N2DH_GOWT1(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip"


class Fluo_N2DL_HeLa(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip"


class PhC_C2DH_U373(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip"


class PhC_C2DL_PSC(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/PhC-C2DL-PSC.zip"


class Fluo_N2DH_SIM(CellTrackingChallengeDataset):
    download_link = "http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip"
