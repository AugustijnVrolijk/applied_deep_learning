# task.py
# Evaluates a trained CIFAR-100 model on a noisy test set
# and saves a montage of 16 MixUp-processed images.

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from train import Net, MixUp, config_cuda

def add_gaussian_noise(inputs, std=0.15):
    """
    Adds Gaussian noise to the input tensor.

    inputs:
        inputs (torch.Tensor): The input tensor to add noise to
        std (float): The standard deviation of the Gaussian noise

    outputs:
        noisy (torch.Tensor): The noisy input tensor
    """
    noisy = inputs + std * torch.randn_like(inputs)
    return torch.clamp(noisy, -1.0, 1.0)

def make_testloader(batch_size: int, use_cuda: bool) -> DataLoader:
    """
    Creates the CIFAR-100 test dataloader with the same normalisation
    used during training.
    inputs:
        batch_size (int): The batch size to use for the dataloader
        use_cuda (bool): Whether to use CUDA for data loading
    
    outputs:
        testloader (DataLoader): A DataLoader for the CIFAR-100 test set
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

    grid = make_grid(mixed_inputs,nrow=4,padding=2,)

    save_image(grid, save_path)
    print(f"Saved MixUp montage to {save_path}")


if __name__ == "__main__":
    """
    main function to evaluate the trained model on a noisy test set and save a MixUp demo image.
      Also includes discussion points on the effects of MixUp and label smoothing on memorization and regularization.
    """
    batch_size = 128
    noise_std = 0.05
    model_path = "best_model_task2.pt"

    device, use_cuda = config_cuda()

    net = Net().to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)

    testloader = make_testloader(batch_size=batch_size, use_cuda=use_cuda)
    noisy_accuracy = evaluate_noisy_testset(net=net, testloader=testloader, device=device, use_cuda=use_cuda,noise_std=noise_std,)

    print(f"Noisy test accuracy (std={noise_std:.2f}): {noisy_accuracy * 100:.2f}%")

    save_mixup_demo(lambda_param=0.4)

    """
    DISCUSS 3 THINGS; Memorization was massively protected with MixUp
    DEFEND MY LABEL SMOOTHING; for my version its mathematically identical to do before or after MixUp
    then look at whether it even helped or not, turn label smoothing off and see if it makes a difference,
    IF NOT THEN MENTION THAT MAYBE MIXUP WAS ALREADY DOING A SIMILAR JOB IN REGULARIZATION, TURNING THE OUTPUT LABELS
    ALREADY INTO SOFT TARGETS; SO THE LABEL SMOOTHING WASN'T SO USEFUL
    """


    """with mixup and label smoothing, got a test accuracy of 51.58%, final loss was 2.43; mean gap was 0.13"""
    """with mixup and no label smoothing, got a test accuracy of 51.04%, final loss was 1.89"""
    """with no mixup and label smoothing, got a test accuracy of 49.40%, final loss was 2.51"""
    """with no mixup and no label smoothing, got a test accuracy of 45.20%, final loss was 2.22; mean gap was 0.52"""