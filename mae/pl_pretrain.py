"""
run the pytorch lightning model
"""
from lightning.callbacks import ModelCheckpoint

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
import lightning as L
import timm

import util.misc as misc
from dataset_mae import MAEDataset
from mae.pl_models_mae import MAEPLRobotLang
assert timm.__version__ == "0.3.2"  # version check
MEAN_CLIPORT = [0.48145466, 0.4578275, 0.40821073]
STD_CLIPORT = [0.26862954, 0.26130258, 0.27577711]
PATH = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/data_hdf5/exist_dataset_no_aug.hdf5'
SEED = 0
CHECKPOINT = '/jmain02/home/J2AD007/txk47/cxz00-txk47/cliport/checkpoints/mae_pretrain_vit_base.pth'

def get_fix_transform():
    trasform_fix = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_CLIPORT, std=STD_CLIPORT)])
    return trasform_fix

def main():

    # TODO: fix the seed for reproducibility here
    # seed = SEED + misc.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # import dataset
    transform_train = get_fix_transform()
    dataset_train = MAEDataset(transform=transform_train, data_path=args.data_path)
    dataset_vis = Subset(dataset_train, range(10))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        num_workers=2,
        drop_last=True,
        shuffle=True
    )

    data_loader_vis = torch.utils.data.DataLoader(
        dataset_vis,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    # define the model
    model = MAEPLRobotLang()

    # load pretrain model
    misc.dynamic_load_pretrain(model, CHECKPOINT)

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor='train_loss',
        mode='min',
        save_last=False,
        save_on_train_epoch_end=True,
    )

    # trainer
    trainer = L.Trainer(
        accelerator='gpu',
        strategy='DDP',
        check_val_every_n_epoch=50,
        callbacks=[checkpoint_callback],
        max_epochs=400,
        devices=1,
        default_root_dir="debug",
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        precision='16-mixed'
    )

    trainer.fit(model,
                train_dataloaders=data_loader_train,
                val_dataloaders=data_loader_vis)


if __name__ == '__main__':
    main()
