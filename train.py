#%%
## DATASET
from audio_data_pytorch import WAVDataset
from audio_data_pytorch.transforms import Crop
import time, os
import torch
#from vscode_audio import Audio
#Audio(sampled.squeeze().numpy(), 44100)

N_SAMPLES = 2 ** 18

crop = Crop(N_SAMPLES)

dataset = WAVDataset(path="/data/tokui/data/bassmusic/", recursive=True, with_sample_rate=44100, transforms=crop)
print(len(dataset))

nb_train_samples = int(len(dataset) * 0.9)
nb_val_samples = len(dataset) - nb_train_samples

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [nb_train_samples, nb_val_samples])
print(len(train_dataset))

# %%

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.utils.data import DataLoader

from pytorch_lightning.loggers import WandbLogger


from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(1000)
# exp_name = "-".join(
#     ["{}:{}".format(k.split(".")[-1], v) for k, v in args.items()]
#     + [str(int(time.time()))]
# )
project_root = "./tmp/models/{}".format(str(int(time.time())))
os.makedirs(project_root, exist_ok=True)
tensorboard_logger = TensorBoardLogger(save_dir="{}/logs/".format(project_root))
dirpath = "{}/models/".format(project_root)
filename = "{epoch}-{val_loss:.4f}"

#%%
wandb_logger = WandbLogger(project="audio_diffusion", tags=[], log_every_n_steps=10)

trainer = Trainer(
    logger=wandb_logger,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10),
        ModelCheckpoint(
            dirpath=dirpath, filename=filename, monitor="val_loss", save_top_k=-1
        ), 
        # WandbImageCallback(valid_dataset,videoembeds=videoembeds, videopaths=videopaths)
    ],
    gpus=1,
#    strategy="ddp", # "dp"  a batch will be distributed to GPUS  "dp" the dataset itself is distributed
    max_epochs=200,
)

batch_size = 8
num_workers = 5

train_loader = DataLoader(
    train_dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    pin_memory=True,
    drop_last=True,
#    worker_init_fn=worker_init_fn,
)

valid_loader = DataLoader(
    val_dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    pin_memory=True,
    drop_last=False,
#    worker_init_fn=worker_init_fn,
)

#%%
## MODEL

from audio_diffusion_pytorch import AudioDiffusionModel
from audio_diffusion_pytorch import UNet1d, Diffusion, LogNormalDistribution
import torch

# UNet used to denoise our 1D (audio) data
unet = UNet1d(
    in_channels=1,
    patch_size=16,
    channels=128,
    multipliers=[1, 2, 4, 4, 4, 4, 4],

    factors=[4, 4, 4, 2, 2, 2],
    attentions=[False, False, False, True, True, True],
    num_blocks=[2, 2, 2, 2, 2, 2],
    attention_heads=8,
    attention_features=64,
    attention_multiplier=2,
    resnet_groups=8,
    kernel_multiplier_downsample=2,
    kernel_sizes_init=[1, 3, 7],
    use_nearest_upsample=False,
    use_skip_scale=True,
    use_attention_bottleneck=True,
    use_learned_time_embedding=True,
)

diffusion = Diffusion(
    net=unet,
    sigma_distribution=LogNormalDistribution(mean = -3.0, std = 1.0),
    sigma_data=0.1,
    dynamic_threshold=0.95
)

class LightningBase(LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log(
            "train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss": loss, "log": {"train_loss": loss.detach()}}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss.detach(), prog_bar=True)
        return {
            "avg_val_loss": avg_loss,
            "log": {"val_loss": avg_loss.detach()},
            "progress_bar": {"val_loss": avg_loss},
        }

    # def test_step(self, batch, batch_idx):
    #     loss = self.step(batch, batch_idx)
    #     return {"test_loss": loss}

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
    #     self.log("test_loss", avg_loss.detach(), prog_bar=True)
    #     return {
    #         "avg_test_loss": avg_loss,
    #         "log": {"test_loss": avg_loss.detach()},
    #         "progress_bar": {"test_loss": avg_loss},
    #     }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam( self.parameters(), lr=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

class DiffusionModel(LightningBase):
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)
        return output       

    def step(self, batch, batch_idx):
        audio, _ = batch
        audio = audio.mean(axis=1, keepdim=True)
        loss = self.forward(audio)
        return loss

diff_model = DiffusionModel(diffusion)

# logging
wandb_logger.watch(diff_model, log="parameters")

#%%

# training
trainer.fit(diff_model, train_loader, valid_loader)
