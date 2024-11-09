
"""
modified from voltron: harness.py

Class defining the evaluation harness for the OCID-Ref Referring Expression Grounding task; a ReferDetectionHarness is
comprised of the following three parts:
    1) __init__  :: Takes backbone, factory function for extractor, factory function for adapter (as LightningModule)
    2) fit       :: Invokes train/fit protocol; for the detection task, this is a traditional supervised learning flow.
                    Uses a Trainer on top of the defined LightningModule --> simple calls to Trainer.fit().
    3) test      :: Function defining the testing (or test metric aggregation) procedure.

By default, assumes a simple MLP bounding-box predictor atop a single (fused) representation; override as you see fit!
"""
import json
import logging
import os
from pathlib import Path
from typing import Callable, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou
from torch.optim import AdamW, Optimizer
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from voltron_evaluation.langref.preprocessing import build_datamodule
from voltron_evaluation.util import LOG_CONFIG, set_global_seed

from transformers import AutoTokenizer

# Grab Logger
logging.config.dictConfig(LOG_CONFIG)
overwatch = logging.getLogger(__file__)

def generate_token(text_processor, lang, device, max_length):
    processed_lang = text_processor(text=lang, padding="max_length", return_tensors='pt', max_length=max_length, truncation=True)
    processed_lang = processed_lang.to(device)
    return processed_lang

class ReferDetectionHarness:
    def __init__(
        self,
        model_id: str,
        backbone: nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        extractor_init_fn: Callable[[], nn.Module],
        detector_init_fn: Callable[[nn.Module, nn.Module], LightningModule] = None,
        run_dir: Path = Path("/home/a/acw694/CLIPort_new_loss/langref"),
        data: Path = Path("/home/a/acw694/datasets/ocid"),
        bsz: int = 512,
        epochs: int = 10,
        seed: int = 7,
        args: Any = None
    ) -> None:
        overwatch.info("Initializing ReferDetectionHarness")
        self.model_id, self.backbone, self.preprocess = model_id, backbone, preprocess
        self.run_dir, self.data, self.bsz, self.epochs, self.seed = run_dir, data, bsz, epochs, seed
        self.extractor_init_fn = extractor_init_fn
        
        # Set Randomness
        set_global_seed(self.seed)

        # Create Run Directory
        os.makedirs(self.run_dir / self.model_id, exist_ok=True)
        self.args = args

    def get_datamodule(self) -> LightningDataModule:
        return build_datamodule(self.data, self.bsz, self.preprocess)

    def fit(self) -> None:
        overwatch.info("Invoking ReferDetectionHarness.fit()")

        # Instantiate DataModule
        overwatch.info("Starting Dataset Processing")
        dm = self.get_datamodule()

        # Create Adapter Model & Callbacks
        overwatch.info("Instantiating Adapter Model and Callbacks")
        detector =DetectorMLP(self.backbone, self.extractor_init_fn(), self.args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(self.run_dir / self.model_id),
            filename="{epoch:02d}-{val_loss:0.4f}-{total_acc25:0.4f}.pt",
            monitor="total_acc25",
            mode="max",
            save_top_k=1,
        )

        overwatch.info("Training...")
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_epochs=self.epochs,
            log_every_n_steps=-1,
            logger=None,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(detector, datamodule=dm)

        # Get final test metrics & serialize...
        test_metrics = trainer.test(detector, datamodule=dm, ckpt_path="best")
        with open(self.run_dir / self.model_id / "final-metrics.json", "w") as f:
            json.dump(
                {
                    "total_acc25": test_metrics[0]["test_total_acc25"],
                    "free_acc25": test_metrics[0]["test_free_acc25"],
                    "touching_acc25": test_metrics[0]["test_touching_acc25"],
                    "stacked_acc25": test_metrics[0]["test_stacked_acc25"],
                },
                f,
                indent=4,
            )

    def test(self) -> None:
        overwatch.info("Compiling Refer Detection Test Metrics")
        with open(self.run_dir / self.model_id / "final-metrics.json", "r") as f:
            metrics = json.load(f)

        # Print Test Metrics!
        overwatch.info("Referring Expression Grounding =>> Test Metrics")
        for mname, mkey in [
            ("Total Accuracy", "total_acc25"),
            ("Accuracy on Free Split (Easy)", "free_acc25"),
            ("Accuracy on Touching Split (Medium)", "touching_acc25"),
            ("Accuracy on Stacked Split (Hard)", "stacked_acc25"),
        ]:
            overwatch.info(f"\t{mname}: {metrics[mkey]:0.4f}")



class DetectorMLP(LightningModule):
    def __init__(
        self, backbone: nn.Module, 
        extractor: nn.Module, 
        args, 
        mlp_features: Tuple[int, ...] = (512, 256, 128, 64)
    ) -> None:
        super().__init__()
        self.mlp_features = [extractor.embed_dim, *list(mlp_features)]

        # Create Network --> Extractor into a "single-shot" detection MLP
        self.backbone, self.extractor, _layers = backbone, extractor, []
        for in_feats, out_feats in zip(self.mlp_features[:-1], self.mlp_features[1:]):
            _layers.append(nn.Linear(in_feats, out_feats))
            _layers.append(nn.GELU())

        # Add final projection =>> xywh bbox coordinates
        _layers.append(nn.Linear(self.mlp_features[-1], 4))
        self.mlp = nn.Sequential(*_layers)
        self.text_processor = AutoTokenizer.from_pretrained(args.text_model)


    def forward(self, img: torch.Tensor, lang: Tuple[str]) -> torch.Tensor:
        # Run through Backbone --> [bsz, n_patches, embed_dim]
        processed_lang = generate_token(self.text_processor, lang, img.device, max_length=77)
        with torch.no_grad():
            patches = self.backbone.forward_refer(img, processed_lang)
        
        # Extract Features --> Detector MLP
        extracted = self.extractor(patches)
        return self.mlp(extracted)

    def training_step(self, batch: Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Unpack batch of RGB frame, language string, bbox (xywh), clutter split, run detector, compute loss."""
        img, lang, bbox, _ = batch

        # Run forward pass, get predicted bbox coordinates =>> [bsz, 4]
        bbox_coords = self.forward(img, lang)

        # Compute huber loss (smooth L1) relative to ground-truth bbox coordinates
        return F.huber_loss(bbox_coords, bbox.float())

    def validation_step(
        self, batch: Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Unpack batch =>> compute Acc @ 0.25 IoU (total, per-split) as the key evaluation metric."""
        img, lang, bbox, clutter_split = batch

        # Run forward pass, get predicted bbox coordinates =>> [bsz, 4]
        bbox_coords = self.forward(img, lang)

        # Compute huber loss (smooth L1) relative to ground-truth bbox coordinates
        loss = F.huber_loss(bbox_coords, bbox.float())

        # Compute various Acc @ 0.25 IoU Metrics --> for *all data* and for each individual split...
        #   => Convert from xywh -> xyxy then use torchvision.ops.box_iou().diagonal() to compute IoU per example
        #   => Threshold based on IoU of 0.25 --> use to compute total accuracy...
        iou = box_iou(box_convert(bbox_coords, "xywh", "xyxy"), box_convert(bbox, "xywh", "xyxy")).diagonal()
        iou_at_25 = iou > 0.25

        # Total Acc @ 0.25
        total_acc25, n_total = iou_at_25.float().sum(), (clutter_split != -1).float().sum()

        # Compute Acc @ 0.25 for each of the three splits...
        free_mask = clutter_split == 0
        free_acc25, n_free = (iou_at_25 & free_mask).float().sum(), free_mask.sum()

        touching_mask = clutter_split == 1
        touching_acc25, n_touching = (iou_at_25 & touching_mask).float().sum(), touching_mask.sum()

        stacked_mask = clutter_split == 2
        stacked_acc25, n_stacked = (iou_at_25 & stacked_mask).float().sum(), stacked_mask.sum()

        return loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked

    def validation_epoch_end(self, step_outputs: List[Tuple[torch.Tensor, ...]]) -> None:
        """Aggregate and log validation metrics."""
        val_loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked = [
            torch.stack(output) for output in zip(*step_outputs)
        ]

        # Reduce & Log...
        self.log_dict(
            {
                "val_loss": val_loss.mean(),
                "total_acc25": total_acc25.sum() / n_total.sum(),
                "free_acc25": free_acc25.sum() / n_free.sum(),
                "touching_acc25": touching_acc25.sum() / n_touching.sum(),
                "stacked_acc25": stacked_acc25.sum() / n_stacked.sum(),
            },
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, str, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, ...]:
        """Unpack batch =>> compute Acc @ 0.25 IoU (total, per-split) as the key evaluation metric."""
        img, lang, bbox, clutter_split = batch

        # Run forward pass, get predicted bbox coordinates =>> [bsz, 4]
        bbox_coords = self.forward(img, lang)

        # Compute huber loss (smooth L1) relative to ground-truth bbox coordinates
        loss = F.huber_loss(bbox_coords, bbox.float())

        # Compute various Acc @ 0.25 IoU Metrics --> for *all data* and for each individual split...
        #   => Convert from xywh -> xyxy then use torchvision.ops.box_iou().diagonal() to compute IoU per example
        #   => Threshold based on IoU of 0.25 --> use to compute total accuracy...
        iou = box_iou(box_convert(bbox_coords, "xywh", "xyxy"), box_convert(bbox, "xywh", "xyxy")).diagonal()
        iou_at_25 = iou > 0.25

        # Total Acc @ 0.25
        total_acc25, n_total = iou_at_25.float().sum(), (clutter_split != -1).float().sum()

        # Compute Acc @ 0.25 for each of the three splits...
        free_mask = clutter_split == 0
        free_acc25, n_free = (iou_at_25 & free_mask).float().sum(), free_mask.sum()

        touching_mask = clutter_split == 1
        touching_acc25, n_touching = (iou_at_25 & touching_mask).float().sum(), touching_mask.sum()

        stacked_mask = clutter_split == 2
        stacked_acc25, n_stacked = (iou_at_25 & stacked_mask).float().sum(), stacked_mask.sum()

        return loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked

    def test_epoch_end(self, step_outputs: List[Tuple[torch.Tensor, ...]]) -> None:
        """Aggregate and log test metrics."""
        test_loss, total_acc25, n_total, free_acc25, n_free, touching_acc25, n_touching, stacked_acc25, n_stacked = [
            torch.stack(output) for output in zip(*step_outputs)
        ]

        # Reduce & Log...
        self.log_dict(
            {
                "test_loss": test_loss.mean(),
                "test_total_acc25": total_acc25.sum() / n_total.sum(),
                "test_free_acc25": free_acc25.sum() / n_free.sum(),
                "test_touching_acc25": touching_acc25.sum() / n_touching.sum(),
                "test_stacked_acc25": stacked_acc25.sum() / n_stacked.sum(),
            },
            prog_bar=True,
        )

    def configure_optimizers(self) -> Optimizer:
        return AdamW([p for p in self.parameters() if p.requires_grad])


def instantiate_detector(backbone: nn.Module, extractor: nn.Module) -> LightningModule:
    return DetectorMLP(backbone, extractor)
