import torch
import torchvision.transforms as transforms
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import torchvision
import torch.utils.data as data
import os

PREPROCESSINGS = {
    'Res256Crop224': transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()]),
    'Crop288': transforms.Compose([transforms.CenterCrop(288),
                                   transforms.ToTensor()]),
    'none': transforms.Compose([transforms.ToTensor()]),
}

def complete_data_dir_path(data_root_dir: str, dataset_name: str):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet-C",
               "imagenet_r": "imagenet-r",
               "imagenet_a": "imagenet-a",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_v2": os.path.join("IamgeNet-V2", "imagenetv2-matched-frequency-format-val"),
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "domainnet126": "DomainNet-126", # directory containing the 6 splits of "cleaned versions" from http://ai.bu.edu/M3SDA/#dataset
               "cifar10": "",       # do not change
               "cifar10_c": "",     # do not change
               "cifar100": "",      # do not change
               "cifar100_c": "",    # do not change
               "caltech101": os.path.join("caltech101", "101_ObjectCategories"),
               "dtd": os.path.join("dtd", "dtd", "images"),
               "eurosat": os.path.join("eurosat", "2750"),                      # automatic download fails
               "fgvc_aircraft": os.path.join("fgvc-aircraft-2013b", "data"),    # do not add 'images' in path
               "flowers102": os.path.join("flowers-102", "jpg"),
               "food101": os.path.join("food-101", "images"),
               "oxford_pets": os.path.join("oxford-iiit-pet", "images"),
               "stanford_cars": os.path.join("stanford_cars"),                  # automatic download fails
               "sun397": os.path.join("sun397"),                                # automatic download fails
               "ucf101": os.path.join("ucf101", "UCF-101-midframes"),           # automatic download fails
               "ccc": "",
               }
    assert dataset_name in mapping.keys(),\
        f"Dataset '{dataset_name}' is not supported! Choose from: {list(mapping.keys())}"
    return os.path.join(data_root_dir, mapping[dataset_name])

def load_imagenet_others(
        batch_size: Optional[int] = 128,
        data_dir: str = './data',
        dataset_name: str = 'imagenet_r',
        shuffle: bool = False,
        prepr: str = 'Res256Crop224'
):
    data_dir = complete_data_dir_path(data_dir, dataset_name)
    transforms_test = PREPROCESSINGS[prepr]
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transforms_test)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=30)

    return test_loader

