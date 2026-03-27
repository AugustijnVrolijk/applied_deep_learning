# task.py
# Evaluates a trained CIFAR-100 model on a noisy test set
# and saves a montage of 16 MixUp-processed images.

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from network import Net
from train import MixUp, config_cuda

def add_gaussian_noise(inputs, std=0.15):
    noisy = inputs + std * torch.randn_like(inputs)
    return torch.clamp(noisy, -1.0, 1.0)

def make_testloader(batch_size: int, use_cuda: bool) -> DataLoader:
    """
    Creates the CIFAR-100 test dataloader with the same normalisation
    used during training.
    """
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])
    testset = torchvision.datasets.CIFAR100(root="./data",train=False,download=True,transform=transform,)
    testloader = DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=2,pin_memory=use_cuda,)

    return testloader

def evaluate_noisy_testset(net: Net,testloader: DataLoader,device: torch.device,use_cuda: bool,noise_std: float = 0.15,) -> float:
    """
    Evaluates classification accuracy on a noisy version of the CIFAR-100 test set.
    
    inputs:
        net (Net): The trained model to evaluate
        testloader (DataLoader): A DataLoader for the CIFAR-100 test set
        device (torch.device): The device to run the evaluation on
        use_cuda (bool): Whether to use CUDA for data loading
        noise_std (float): The standard deviation of the Gaussian noise to add to the inputs

    outputs:
        accuracy (float): The classification accuracy on the noisy test set, as a value between
    """
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            noisy_inputs = add_gaussian_noise(inputs, std=noise_std)

            outputs = net(noisy_inputs)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy

def save_mixup_demo(lambda_param: float = 0.4) -> None:
    """
    Saves a 4x4 montage of 16 images after applying MixUp.

    Uses the first 16 CIFAR-100 training images so the montage is deterministic
    apart from the sampled MixUp lambda/permutation.

    inputs:
        lambda_param (float): The mixing coefficient to use for the MixUp augmentation. Should be between 0 and 1

    outputs:
        None (saves an image file "robustness_demo.png" in the current working directory
    """
    save_path = "robustness_demo.png"

    batch_size = 16 # 16 image montage, so batch size of 16 to get one batch of mixed images
    testloader = make_testloader(batch_size=batch_size, use_cuda=False) # just to get the number of classes for MixUp, not actually used for testing here
    
    num_classes = len(testloader.dataset.classes)
    mixup = MixUp(alpha=0.5, num_classes=num_classes) # alpha is ignored, set determined lambda_param

    inputs, labels = next(iter(testloader))
    mixed_inputs, _ = mixup(inputs, labels, lambda_param = 0.4)

    # unnormalise the mixed inputs for saving as an image
    mixed_inputs = mixed_inputs * 0.5 + 0.5
    mixed_inputs = torch.clamp(mixed_inputs, 0.0, 1.0)

    grid = make_grid(
        mixed_inputs,
        nrow=4,
        padding=2,
    )

    save_image(grid, save_path)
    print(f"Saved MixUp montage to {save_path}")


if __name__ == "__main__":
    batch_size = 128
    noise_std = 0.05
    model_path = "best_model.pt"

    device, use_cuda = config_cuda()

    net = Net().to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)

    testloader = make_testloader(batch_size=batch_size, use_cuda=use_cuda)
    noisy_accuracy = evaluate_noisy_testset(net=net, testloader=testloader, device=device, use_cuda=use_cuda,noise_std=noise_std,)

    print(f"Noisy test accuracy (std={noise_std:.2f}): {noisy_accuracy * 100:.2f}%")

    save_mixup_demo(lambda_param=0.4)