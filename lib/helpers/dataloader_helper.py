import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from torch.utils.data import ConcatDataset

# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, workers=8):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        # train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        train_set_list = []
        for subset_name in cfg["train_subsets"]:
            train_set_list.append(KITTI_Dataset(split=cfg['train_split'], cfg=cfg, subset_train=subset_name))
        train_set = ConcatDataset(train_set_list)
        val_subset_name = cfg["val_subset"]
        test_set = KITTI_Dataset(split=cfg['test_split'], cfg=cfg, subset_val=val_subset_name)
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    # resolve num_workers from cfg if provided
    workers = cfg.get('num_workers', workers)

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=workers,
                              worker_init_fn=my_worker_init_fn,
                              shuffle=True,
                              pin_memory=False,
                              drop_last=cfg.get('drop_last', True))
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=workers,
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader
