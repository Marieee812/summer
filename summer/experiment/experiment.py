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


class Experiment(ExperimentBase):
    def __init__(
        self, model_checkpoint: Optional[Path] = None, test_dataset: Optional[CellTrackingChallengeDataset] = None
    ):
        if model_checkpoint is not None:
            assert model_checkpoint.exists(), model_checkpoint
            assert model_checkpoint.is_file(), model_checkpoint

        self.model = UNet(in_channels=1, n_classes=2, depth=5, wf=6, padding=True, batch_norm=False, up_mode="upsample")

        self.train_dataset = DIC_C2DH_HeLa(one=True, two=False, labeled_only=True, transform=self.train_transform)
        eval_ds = DIC_C2DH_HeLa(one=False, two=True, labeled_only=True, transform=self.eval_transform)
        self.valid_dataset = eval_ds
        self.test_dataset = eval_ds if test_dataset is None else test_dataset
        self.max_validation_samples = 3

        self.batch_size = 1
        self.eval_batch_size = 6
        self.precision = torch.float
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer_cls = torch.optim.Adam
        self.optimizer_kwargs = {"lr": 1e-4, "eps": eps_for_precision[self.precision]}
        self.max_num_epochs = 10

        self.model_checkpoint = model_checkpoint

        super().__init__()

    def train_transform(self, img: Image.Image, seg: Image.Image) -> Tuple[torch.Tensor, torch.LongTensor]:
        # Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270 or Image.TRANSPOSE
        img, seg = TF.to_tensor(img), TF.to_tensor(seg)
        assert seg.shape[0] == 1, seg.shape  # assuming singleton channel axis
        seg = (seg[0] != 0)
        return img.to(dtype=self.precision), seg.to(dtype=torch.long)

    def eval_transform(self, img: Image.Image, seg: Image.Image) -> Tuple[torch.Tensor, torch.LongTensor]:
        img, seg = TF.to_tensor(img), TF.to_tensor(seg)
        return img.to(dtype=self.precision), (seg != 0).to(dtype=torch.long)

    def score_function(self, engine):
        val_loss = engine.state.metrics[LOSS_NAME]
        return -val_loss


def run():
    assert torch.cuda.device_count() <= 1, "visible cuda devices not limited!"
    exp = Experiment()
    exp.run()
