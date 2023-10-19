import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def plot_images(images: torch.Tensor):
    """
    Create a small image grid from a tensor and plots them.

    Args:
        images (torch.Tensor): the tensor containing the images
    """
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images: torch.Tensor, path: str, **kwargs):
    """
    Create a small image grid from a tensor and saves it to the disk.

    Args:
        images (torch.Tensor): the tensor containing the images
        path (str): path to save the images to
    """
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args: dict) -> DataLoader:
    """
    Creates a `DataLoader` from a directory containing any images.

    Args:
        args (dict): the run arguments

    Returns:
        DataLoader: a DataLoader containing the training images ready to be fed to the model
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def save_image_grid(dir_path: str, img_size: int=64, grid_name: str="samples.jpg") -> None:
    """
    Create a collage of images located in a directory and saves it in the same directory.

    Args:
        dir_path (str): path to the directory containing the images
        img_size (int, optional): size (in pixels) of the images. Defaults to 64.
        grid_name (str, optional): file name under which the final grid will be saved. Defaults to "samples.jpg".
    """
    image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
    num_images = len(image_files)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid_width = num_cols * img_size
    grid_height = num_rows * img_size
    grid = Image.new('RGB', (grid_width, grid_height))
    x, y = 0, 0
    for image_file in image_files:
        img = Image.open(os.path.join(dir_path, image_file))
        grid.paste(img, (x, y))
        x += img_size
        if x >= grid_width:
            x = 0
            y += img_size
    grid.save(os.path.join(dir_path, grid_name))

def setup_logging(run_name: str):
    """
    Initializes the necessary directories before a run.

    Args:
        run_name (str): name of the run
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)