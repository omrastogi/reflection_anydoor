import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

import sys
sys.path.append('/data/om/reflection_anydoor/datasets')
sys.path.append('/data/om/reflection_anydoor')

from mirrors import MirrorsDataset
from mirrors_couterfactual import MirrorsCounterfactualDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = '/data/om/reflection_anydoor/lightning_logs/version_24/checkpoints/epoch=999-step=238999.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches=1

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/anydoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
DConf = OmegaConf.load('./configs/datasets.yaml')

import wandb
from datetime import datetime
wandb.login(key='39e65ab86c39c92f1b18458c6cc56fee691e0705')
wandb.init(project="anydoor_inpainting")
wandb.run.name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
wandb_logger = WandbLogger()

dataset = MirrorsCounterfactualDataset(**DConf.Train.MirrorsCounterfactual)
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(
    gpus=[0], 
    strategy="ddp", 
    precision=16, 
    accelerator="gpu", 
    progress_bar_refresh_rate=1, 
    accumulate_grad_batches=accumulate_grad_batches,
    logger=wandb_logger,
    )

# Train!
trainer.fit(model, dataloader)
