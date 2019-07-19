import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy
import pbs3
import torch.nn

from datetime import datetime
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.metrics import Loss, Accuracy
from ignite.utils import convert_tensor
from inferno.io.transform import Transform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from typing import Union, Optional, List, Dict, Callable, Type, Any, Tuple, Sequence
from PIL import Image

from summer.datasets import CellTrackingChallengeDataset

eps_for_precision = {torch.half: 1e-4, torch.float: 1e-8}

IN_TRAINING = "in_training"
TRAINING = "training"
VALIDATION = "validation"
TEST = "test"

X_NAME = "x"
Y_NAME = "y"
Y_PRED_NAME = "y_pred"
LOSS_NAME = "Loss"
ACCURACY_NAME = "Accuracy"


@dataclass
class LogConfig:
    log_dir: Path = Path(os.environ.get("LOG_DIR", Path(__name__).parent.parent.parent / "logs"))

    validate_every_nth_epoch: int = 1
    log_scalars_every: Tuple[int, str] = (1, "iterations")
    log_images_every: Tuple[int, str] = (1, "epochs")

    # send_image_at_batch_indices: Optional[Tuple[int]] = None
    # send_image_at_z_indices: Optional[Tuple[int]] = (0, 5, 10, -1)
    # send_image_at_channel_indices: Optional[Tuple[int]] = (0,)


class ExperimentBase:
    model: torch.nn.Module

    train_dataset: CellTrackingChallengeDataset
    valid_dataset: CellTrackingChallengeDataset
    test_dataset: Optional[CellTrackingChallengeDataset]
    max_validation_samples: int

    batch_size: int
    eval_batch_size: int
    precision: torch.dtype
    loss_fn: torch.nn.Module
    optimizer_cls: Type[torch.optim.Optimizer]
    optimizer_kwargs: Dict[str, Any]
    max_num_epochs: int

    model_checkpoint: Optional[Path]

    score_function: Callable[[Engine], float]

    def __init__(self):
        self.log_config = LogConfig()
        assert self.log_config.log_scalars_every[1] in ("iterations", "epochs"), self.log_config.log_scalars_every[1]
        assert self.log_config.log_images_every[1] in ("iterations", "epochs"), self.log_config.log_images_every[1]

        self.commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
        self.commit_subject = pbs3.git.log("-1", "--pretty=%B").stdout.split("\n")[0]
        self.branch_name = (
            pbs3.git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip().replace("'", "").replace('"', "")
        )
        if self.valid_dataset == self.test_dataset and self.max_validation_samples >= len(self.test_dataset):
                raise ValueError("no samples for testing left")

    def test(self):
        self.max_num_epochs = 0
        self.run()

    def run(self):
        short_commit_subject = (
            self.commit_subject[5:15].replace(":", "").replace("'", "").replace('"', "").replace(" ", "_")
        )
        self.name = f"{datetime.now().strftime('%y-%m-%d_%H-%M')}_{self.commit_hash[:7]}_{self.branch_name}_{short_commit_subject}"
        self.logger = logging.getLogger(self.name)
        self.log_config.log_dir /= self.name
        self.log_config.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_config.log_dir / "commit_hash").open("w") as f:
            f.write(self.commit_hash)

        devices = list(range(torch.cuda.device_count()))
        if devices:
            device = torch.device("cuda", devices[0])
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device=device, dtype=self.precision)
        if self.model_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.model_checkpoint, map_location=device))

        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=16, shuffle=True
        )
        train_loader_eval = DataLoader(
            self.train_dataset,
            batch_size=self.eval_batch_size,
            pin_memory=True,
            num_workers=16,
            sampler=SubsetSequentialSampler(range(min(200, len(self.train_dataset)))),
        )
        valid_loader = (
            None
            if self.valid_dataset is None
            else DataLoader(
                self.valid_dataset,
                batch_size=self.eval_batch_size,
                pin_memory=True,
                num_workers=16,
                sampler=SubsetSequentialSampler(range(min(self.max_validation_samples, len(self.valid_dataset)))),
            )
        )
        if self.valid_dataset == self.test_dataset:
            test_sampler = SubsetSequentialSampler(range(self.max_validation_samples, len(self.test_dataset)))
        else:
            test_sampler = SubsetSequentialSampler(range(len(self.test_dataset)))

        test_loader = (
            None
            if self.test_dataset is None
            else DataLoader(
                self.test_dataset,
                batch_size=self.eval_batch_size,
                pin_memory=True,
                num_workers=16,
                sampler=test_sampler,
            )
        )

        # tensorboardX
        writer = SummaryWriter(log_dir=self.log_config.log_dir.as_posix())
        # data_loader_iter = iter(train_loader)
        # x, y = next(data_loader_iter)
        # try:
        #     writer.add_graph(self.model, x.to(torch.device("cuda")))
        # except Exception as e:
        #     self.logger.warning("Failed to save model graph...")
        #     self.logger.exception(e)

        # ignite
        def training_step(engine, batch):
            self.model.train()
            optimizer.zero_grad()
            x, y = batch
            x = convert_tensor(x, device=device, non_blocking=False)
            y = convert_tensor(y, device=device, non_blocking=False)
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            return {X_NAME: x, Y_NAME: y, Y_PRED_NAME: y_pred, LOSS_NAME: loss.item()}

        trainer = Engine(training_step)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        def inference_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch
                x = convert_tensor(x, device=device, non_blocking=False)
                y = convert_tensor(y, device=device, non_blocking=False)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                return {X_NAME: x, Y_NAME: y, Y_PRED_NAME: y_pred, LOSS_NAME: loss.item()}

        class EngineWithMode(Engine):
            def __init__(self, process_function, modes: Sequence[str]):
                super().__init__(process_function=process_function)
                self.modes = modes
                self._mode = None

            @property
            def mode(self) -> str:
                if self._mode is None:
                    raise RuntimeError("mode not set")

                return self._mode

            @mode.setter
            def mode(self, new_mode: str):
                if new_mode not in self.modes:
                    raise ValueError(new_mode)
                else:
                    self._mode = new_mode

        evaluator = EngineWithMode(inference_step, modes=[TRAINING, VALIDATION, TEST])
        evaluator.mode = TRAINING
        saver = ModelCheckpoint(
            (self.log_config.log_dir / "models").as_posix(),
            "v0",
            score_function=self.score_function,
            n_saved=1,
            create_dir=True,
            save_as_state_dict=True,
        )
        evaluator.add_event_handler(Events.COMPLETED, saver, {"model": self.model})
        stopper = EarlyStopping(patience=10, score_function=self.score_function, trainer=trainer)
        evaluator.add_event_handler(Events.COMPLETED, stopper)

        Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out: (out[LOSS_NAME], out[X_NAME])).attach(
            evaluator, LOSS_NAME
        )
        Accuracy(output_transform=lambda out: (out[Y_PRED_NAME], out[Y_NAME])).attach(evaluator, ACCURACY_NAME)

        result_saver = ResultSaver(TEST, self.log_config.log_dir / "test-result", "seg")
        @evaluator.on(Events.ITERATION_COMPLETED)
        def export_result(engine: EngineWithMode):
            result_saver.save(engine.mode, batch=engine.state.output[Y_PRED_NAME], at=engine.state.iteration - 1)

        def log_scalars(engine: Engine, name: str, step: int):
            met = engine.state.metrics
            writer.add_scalar(f"{name}/Loss", met[LOSS_NAME], step)
            writer.add_scalar(f"{name}/Accuracy", met[ACCURACY_NAME], step)

        def log_images(engine: Engine, name: str, step: int):
            x_batch = engine.state.output[X_NAME].cpu().numpy()
            y_batch = engine.state.output[Y_NAME].cpu().numpy()
            y_pred_batch = engine.state.output[Y_PRED_NAME].detach().cpu().numpy()
            assert x_batch.shape[0] == y_batch.shape[0], (x_batch.shape, y_batch.shape)
            assert len(y_batch.shape) == 5, y_batch.shape
            assert y_batch.shape[1] == 1, y_batch.shape

            fig, ax = plt.subplots(
                nrows=x_batch.shape[0], ncols=4, squeeze=False, figsize=(4 * 3, x_batch.shape[0] * 3)
            )
            fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)

            def make_subplot(ax, img):
                im = ax.imshow(img)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis("off")
                # from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                # create an axes on the right side of ax. The width of cax will be 5%
                # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.03)
                fig.colorbar(im, cax=cax)

            for i, (xx, yy, pp) in enumerate(zip(x_batch, y_batch, y_pred_batch)):
                if i == 0:
                    ax[0, 0].set_title("input")
                    ax[0, 1].set_title("target")
                    ax[0, 2].set_title("output")
                    ax[0, 3].set_title("diff")

                make_subplot(ax[i, 0], xx[0])
                make_subplot(ax[i, 1], yy[0])
                make_subplot(ax[i, 2], pp[0])
                make_subplot(ax[i, 3], (pp == yy)[0])

            plt.tight_layout()

            writer.add_figure(f"{name}/in_out", fig, step)

        def log_eval(engine: Engine, name: str, step: int):
            log_scalars(engine=engine, name=name, step=step)
            log_images(engine=engine, name=name, step=step)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_iteration(engine: Engine):
            i = (engine.state.iteration - 1) % len(train_loader)
            if self.log_config.log_scalars_every[1] == "iterations" and i % self.log_config.log_scalars_every[0] == 0:
                log_scalars(engine, IN_TRAINING, engine.state.iteration)

            if self.log_config.log_images_every[1] == "iterations" and i % self.log_config.log_images_every[0] == 0:
                log_images(engine, IN_TRAINING, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_epoch(engine: Engine):
            epoch = engine.state.epoch
            if self.log_config.log_scalars_every[1] == "epochs" and epoch % self.log_config.log_scalars_every[0] == 0:
                log_scalars(engine, IN_TRAINING, epoch)

            if self.log_config.log_images_every[1] == "epochs" and epoch % self.log_config.log_images_every[0] == 0:
                log_images(engine, IN_TRAINING, epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine: Engine):
            if engine.state.epoch % self.log_config.validate_every_nth_epoch == 0:
                # evaluate on training data
                evaluator.mode = TRAINING
                evaluator.run(train_loader_eval)
                self.logger.info(
                    "Training Results  -  Epoch: %d  Avg loss: %.3f",
                    engine.state.epoch,
                    evaluator.state.metrics[LOSS_NAME],
                )
                log_eval(evaluator, TRAINING, engine.state.epoch)

                # evaluate on validation data
                evaluator.mode = VALIDATION
                evaluator.run(valid_loader)
                self.logger.info(
                    "Validation Results - Epoch: %d  Avg loss: %.3f",
                    engine.state.epoch,
                    evaluator.state.metrics[LOSS_NAME],
                )
                log_eval(evaluator, VALIDATION, engine.state.epoch)

        @trainer.on(Events.COMPLETED)
        def test(engine: Engine):
            evaluator.mode = TEST
            evaluator.run(test_loader)
            self.logger.info(
                "Test Results    -    Epoch: %d  Avg loss: %.3f", engine.state.epoch, evaluator.state.metrics[LOSS_NAME]
            )
            log_eval(evaluator, TEST, engine.state.epoch)

        trainer.run(train_loader, max_epochs=self.max_num_epochs)
        writer.close()


class SubsetSequentialSampler(torch.utils.data.sampler.Sampler):
    """Samples elements in fixed order from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: Sequence[int]):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ResultSaver:
    def __init__(self, *names: str, file_path: Path):
        self.folders = {name: file_path / name for name in names}
        for dir in self.folders.values():
            dir.mkdir(parents=True)

    def save(self, name: str, batch: torch.tensor, at: int):
        if name not in self.folders:
            return

        batch = batch.detach().cpu().numpy()
        assert len(batch.shape) == 4 or len(batch.shape) == 5, batch.shape
        batch = (batch.clip(min=0, max=1) * numpy.iinfo(numpy.uint16).max).astype(numpy.uint16)
        for i, img in enumerate(batch, start=at):
            Image.fromarray(img).save(self.folders[name] / f"seg{i:04}.tif")
