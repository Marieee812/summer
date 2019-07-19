import logging
import torch
import urllib.request

from io import BytesIO
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple, Callable
from zipfile import ZipFile


class CellTrackingChallengeDataset(Dataset):
    download_link: str

    def __init__(
        self,
        one: bool,
        two: bool,
        labeled_only: bool,
        transform: Callable[[Image.Image, Image.Image], Tuple[torch.Tensor, torch.LongTensor]],
    ):
        self.transform = transform
        folder = self.download_link.split("/")[-1].replace(".zip", "/")
        self.logger = logging.getLogger(folder)
        self.path: Path = Path(__file__).parent.parent / "data" / folder
        if not (self.path).exists():
            with urllib.request.urlopen(self.download_link) as response, ZipFile(BytesIO(response.read())) as zipf:
                assert folder in zipf.namelist(), zipf.namelist()
                zipf.extractall(path=self.path.parent)

        # collect valid indices
        self.indices = [[], []]
        self.folder_names = ["01", "02"]
        for name, selected, idx in zip(self.folder_names, [one, two], self.indices):
            if not selected:
                continue

            if labeled_only:
                name += "_GT/SEG"

            folder = self.path / name

            idx += [int(file_name.name[-7:-4]) for file_name in folder.glob("*.tif")]

        self._len = sum([len(idx) for idx in self.indices])

    def __len__(self):
        return self._len

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        original_item = item
        for i, idx in enumerate(self.indices):
            if item < len(idx):
                folder = self.folder_names[i]
                return self.transform(
                    Image.open(self.path / folder / f"t{idx[item]:03}.tif"),
                    Image.open(self.path / f"{folder}_GT/SEG/man_seg{idx[item]:03}.tif"),
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
