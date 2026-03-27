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

def evaluate_noisy_testset(
    net: Net,
    testloader: DataLoader,
    device: torch.device,
    use_cuda: bool,
    noise_std: float = 0.15,
) -> float:
    """
    Evaluates classification accuracy on a noisy version of the CIFAR-100 test set.
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

def unnormalise_images(inputs: torch.Tensor) -> torch.Tensor:
    """
    Reverses the normalisation:
        x_norm = (x - 0.5) / 0.5
    so:
        x = x_norm * 0.5 + 0.5

    multiplies by 255 to convert to pixel values in [0, 255], and clamps to ensure valid pixel range.

    return inputs in the range [0, 255] as uint8 for saving as images
    """
    inputs = inputs * 0.5 + 0.5
    inputs = torch.clamp(inputs, 0.0, 1.0)
    return inputs


def save_mixup_demo(
    alpha: float = 0.4
) -> None:
    """
    Saves a 4x4 montage of 16 images after applying MixUp.

    Uses the first 16 CIFAR-100 training images so the montage is deterministic
    apart from the sampled MixUp lambda/permutation.
    """
    save_path = "robustness_demo.png"

    batch_size = 16 # 16 image montage, so batch size of 16 to get one batch of mixed images
    testloader = make_testloader(batch_size=batch_size, use_cuda=False) # just to get the number of classes for MixUp, not actually used for testing here
    
    num_classes = len(testloader.dataset.classes)
    mixup = MixUp(alpha=alpha, num_classes=num_classes)

    inputs, labels = next(iter(testloader))
    mixed_inputs, _ = mixup(inputs, labels, lambda_param = 0.4)

    mixed_inputs = unnormalise_images(mixed_inputs) # unnormalise for saving as images

    grid = make_grid(
        mixed_inputs,
        nrow=4,
        padding=2,
    )

    save_image(grid, save_path)
    print(f"Saved MixUp montage to {save_path}")


if __name__ == "__main__":
    batch_size = 128
    noise_std = 0.15
    model_path = "best_model.pt"

    device, use_cuda = config_cuda()

    net = Net().to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)

    testloader = make_testloader(batch_size=batch_size, use_cuda=use_cuda)

    noisy_accuracy = evaluate_noisy_testset(
        net=net,
        testloader=testloader,
        device=device,
        use_cuda=use_cuda,
        noise_std=noise_std,
    )

    print(f"Noisy test accuracy (std={noise_std:.2f}): {noisy_accuracy:.4f}")
    print(f"Noisy test accuracy: {noisy_accuracy * 100:.2f}%")

    save_mixup_demo(alpha=0.4)