import numpy
import torch
import torch.nn
import torch.optim
import torch.utils.data
import torchvision.transforms.functional as TF

from pathlib import Path
from PIL import Image
from typing import Optional, Tuple

from summer.datasets import (
    DIC_C2DH_HeLa,
    Fluo_C2DL_MSC,
    Fluo_N2DH_GOWT1,
    Fluo_N2DL_HeLa,
    PhC_C2DH_U373,
    PhC_C2DL_PSC,
    Fluo_N2DH_SIM,
    CellTrackingChallengeDataset,
)
from summer.experiment.base import ExperimentBase, eps_for_precision, LOSS_NAME
from summer.models.unet import UNet
from summer.utils.stat import DatasetStat


class Experiment(ExperimentBase):
    def __init__(
        self, model_checkpoint: Optional[Path] = None, test_dataset: Optional[CellTrackingChallengeDataset] = None
    ):
        if model_checkpoint is not None:
            assert model_checkpoint.exists(), model_checkpoint
            assert model_checkpoint.is_file(), model_checkpoint

        self.depth = 3
        self.model = UNet(
            in_channels=1, n_classes=1, depth=self.depth, wf=6, padding=True, batch_norm=False, up_mode="upsample"
        )

        self.train_dataset = Fluo_C2DL_MSC(one=False, two=True, labeled_only=True, transform=self.train_transform)
        eval_ds = Fluo_N2DL_HeLa(one=True, two=False, labeled_only=True, transform=self.eval_transform)

        self.valid_dataset = eval_ds
        self.test_dataset = eval_ds if test_dataset is None else test_dataset
        self.max_validation_samples = 3

        self.batch_size = 1
        self.eval_batch_size = 1
        self.precision = torch.float
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer_cls = torch.optim.Adam
        self.optimizer_kwargs = {"lr": 1e-3, "eps": eps_for_precision[self.precision]}
        self.max_num_epochs = 1

        self.model_checkpoint = model_checkpoint

        super().__init__()

    def to_tensor(self, img: Image.Image, seg: Image.Image, stat: DatasetStat) -> Tuple[torch.Tensor, torch.Tensor]:
        img: torch.Tensor = TF.to_tensor(img)
        seg: torch.Tensor = TF.to_tensor(seg)
        assert img.shape == seg.shape, (img.shape, seg.shape)
        assert seg.shape[0] == 1, seg.shape  # assuming singleton channel axis
        cut1 = img.shape[1] % 2 ** self.depth
        if cut1:
            img = img[:, cut1 // 2 : -((cut1 + 1) // 2)]
            seg = seg[:, cut1 // 2 : -((cut1 + 1) // 2)]

        cut2 = img.shape[2] % 2 ** self.depth
        if cut2:
            img = img[:, :, cut2 // 2 : -((cut2 + 1) // 2)]
            seg = seg[:, :, cut2 // 2 : -((cut2 + 1) // 2)]

        img = img.clamp(stat.x_min, stat.x_max)
        img = TF.normalize(img, mean=[stat.x_mean], std=[stat.x_std])

        return img.to(dtype=self.precision), (seg[0] != 0).to(dtype=self.precision)

    def train_transform(
        self, img: Image.Image, seg: Image.Image, stat: DatasetStat
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270 or Image.TRANSPOSE
        return self.to_tensor(img, seg, stat)

    def eval_transform(
        self, img: Image.Image, seg: Image.Image, stat: DatasetStat
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.to_tensor(img, seg, stat)

    def score_function(self, engine):
        val_loss = engine.state.metrics[LOSS_NAME]
        return -val_loss


def run():
    assert torch.cuda.device_count() <= 1, "visible cuda devices not limited!"
    exp = Experiment()
    exp.run()
