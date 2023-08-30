from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torchvision.transforms import Resize
from torch.utils.data import Subset
import numpy as np
import math
import tqdm
import torch
import torchvision.transforms as transforms

class MyCIFAR10DVS(Subset):
    def __init__(self, root, train_ratio=0.9, data_type="frame", frames_number=10, split_by="number", random_split=False, size=(48, 48)):
        transform_dvs = transforms.Compose([
                lambda x: torch.from_numpy(x), 
                Resize(size=size)])

        dataset_dvs = CIFAR10DVS(root=root, data_type=data_type, frames_number=frames_number, split_by=split_by, transform=transform_dvs)

        train_idx, test_idx = self.split_to_train_test_set(train_ratio=train_ratio, origin_dataset=dataset_dvs, num_classes=10, random_split=random_split)
        self.train_dvs = train_idx
        self.test_dvs = test_idx
        
    def split(self):
        
        return self.train_dvs, self.test_dvs
        

        # super().__init__(dataset_dvs, train_idx if train_ratio == 0.9 else test_idx)
    
    # def __getitem__(self, index):
    #     data, label = self.dataset[index]
    #     return data, label

    @staticmethod
    def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):

        label_idx = []
        for i in range(num_classes):
            label_idx.append([])

        for i, item in enumerate(tqdm.tqdm(origin_dataset)):
            y = item[1]
            if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
                y = y.item()
            label_idx[y].append(i)
        train_idx = []
        test_idx = []
        if random_split:
            for i in range(num_classes):
                np.random.shuffle(label_idx[i])

        for i in range(num_classes):
            pos = math.ceil(label_idx[i].__len__() * train_ratio)
            train_idx.extend(label_idx[i][0: pos])
            test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

        return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)
    # def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):

    #     label_idx = []
    #     for i in range(num_classes):
    #         label_idx.append([])

    #     for i, item in enumerate(tqdm.tqdm(origin_dataset)):
    #         y = item[1]
    #         if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
    #             y = y.item()
    #         label_idx[y].append(i)
    #     train_idx = []
    #     test_idx = []

    #     for i in range(num_classes):
    #         np.random.shuffle(label_idx[i])  # ensuring random selection even if random_split is False

    #         pos = math.ceil(len(label_idx[i]) * train_ratio)
    #         train_idx.extend(label_idx[i][0: pos])
    #         test_idx.extend(label_idx[i][pos: ])  # making sure there's no overlap

    #     return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)
