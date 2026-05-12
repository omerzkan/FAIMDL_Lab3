

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_transform():

    train_transform = T.Compose([
        T.RandomHorizontalFlip(), 
        T.RandomCrop(64, padding=4),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    ])

    test_transform = T.Compose([
        T.resize((64, 64)),
        T.totensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, test_transform


def get_dataloaders(data_root, batch_size=64, num_workers=4):
    
    train_transform, test_transform = get_transform()
    
    train_dataset = ImageFolder(
        root=f"{data_root}/train",
        transform=train_transform
    )
    
    val_dataset = ImageFolder(
        root=f"{data_root}/val",
        transform = test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
        )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
        )

    return train_loader, val_loader
