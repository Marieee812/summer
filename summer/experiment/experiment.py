import random

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
from summer.experiment.base import ExperimentBase, eps_for_precision, ACCURACY_NAME
from summer.models.unet import UNet
from summer.utils.stat import DatasetStat

# another 
class Experiment(ExperimentBase):
    def __init__(
        self,
        model_checkpoint: Optional[Path] = None,
        test_dataset: Optional[CellTrackingChallengeDataset] = None,
        add_in_name: Optional[str] = None,
    ):
        if model_checkpoint is not None:
            assert model_checkpoint.exists(), model_checkpoint
            assert model_checkpoint.is_file(), model_checkpoint

        self.depth = 3
        self.model = UNet(
            in_channels=1, n_classes=1, depth=self.depth, wf=4, padding=True, batch_norm=True, up_mode="upsample"
        )

        self.train_dataset = Fluo_N2DH_SIM(one=True, two=False, labeled_only=True, transform=self.train_transform)
        self.valid_dataset = Fluo_N2DH_GOWT1(one=True, two=False, labeled_only=True, transform=self.eval_transform)
        test_ds = Fluo_N2DH_GOWT1(one=False, two=True, labeled_only=True, transform=self.eval_transform)
        self.test_dataset = test_ds if test_dataset is None else test_dataset
        self.max_validation_samples = 10
        self.only_eval_where_true = False

        self.batch_size = 1
        self.eval_batch_size = 1
        self.precision = torch.float
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer_cls = torch.optim.Adam
        self.optimizer_kwargs = {"lr": 1e-5, "eps": eps_for_precision[self.precision]}
        self.max_num_epochs = 50

        self.model_checkpoint = model_checkpoint
        self.add_in_name = add_in_name
        super().__init__()

    def train_transform(
        self, img: Image.Image, seg: Image.Image, stat: DatasetStat
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        tmethod = random.choice(
            [
                Image.FLIP_LEFT_RIGHT,
                Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90,
                Image.ROTATE_180,
                Image.ROTATE_270,
                Image.TRANSPOSE,
            ]
        )
        img = img.transpose(tmethod)
        seg = seg.transpose(tmethod)


        img, seg = self.to_tensor(img, seg, stat)

        #if self.precision == torch.half and img.get_device() == -1:
        #     # meager support for cpu half tensor
        #    img = img.to(dtype=torch.float)
        #    img += torch.zeros_like(img).normal_(std=0.1)
        #    img = img.to(dtype=self.precision)
        #else:
        #    img += torch.zeros_like(img).normal_(std=0.1)

        return img, seg

    def eval_transform(
        self, img: Image.Image, seg: Image.Image, stat: DatasetStat
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.to_tensor(img, seg, stat)

    def score_function(self, engine):
        score = engine.state.metrics[ACCURACY_NAME]
        return score


def run(add_in_name: Optional[str] = None):
    assert torch.cuda.device_count() <= 1, "visible cuda devices not limited!"
    exp = Experiment(add_in_name=add_in_name)
    exp.run()
