# training script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#
#
# Gen AI usage statement:
#
# Gen AI was used to create the draw_accuracy_plot function, which is not the main focus of the assignment
# It was also used to help generate typehints and docstrings for functions, which were after validated by me
# The main training loop, logic and functions were written by me, however Gen AI was used to help understand
# the structure of Pytorch, i.e. understanding loss.backward(), requires the criterion function to return
# a tensor object, rather than just performing the math and returning a scalar. It also helped with guidance
# regarding implementing CUDA friendly code to speed up training on my machine, such as getting a device parameter
# and smaller details like non_blocking transfers. 


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import random_split, DataLoader
from torchvision.ops import DropBlock2d
from PIL import Image, ImageDraw, ImageFont
from itertools import product

from typing import Tuple

# sanity checks to make sure my GPU was being used
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))  
# ------------------------------ UTIL ------------------------------- 

def calculate_mean_gap(train_loss: list[float], val_loss: list[float]) -> float:
    """
    Calculate the mean gap between training and validation loss across epochs. Used for the writeup 

    inputs:
        train_loss: list of training losses for each epoch
        val_loss: list of validation losses for each epoch
    outputs:
        mean_gap: The average difference between training and validation loss across epochs
    """
    if len(train_loss) != len(val_loss):
        raise ValueError("train_loss and val_loss must have the same length")

    n_epochs = len(train_loss)
    total_gap = sum(train - val for train, val in zip(train_loss, val_loss))
    mean_gap = total_gap / n_epochs
    return mean_gap

def draw_accuracy_comparison_plot(
    series: list[tuple[str, list[float]]],
    save_path: str = "generalization_gap.png",
    title: str = "Training vs Validation Accuracy"
) -> None:
    """
    Draw a multi-line accuracy plot using Pillow.

    inputs:
        series: list of (label, values) tuples, where each values list contains
                the accuracy for each epoch
        save_path: path to save the generated PNG
        title: plot title

    outputs:
        None
    """

    width = 1000
    height = 650

    if len(series) == 0:
        raise ValueError("series must contain at least one line")

    n_epochs = max(len(values) for _, values in series)

    if n_epochs < 2:
        raise ValueError("Need at least 2 epochs to draw a line plot")

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Margins
    left = 80
    right = 220   # more space for a bigger legend
    top = 60
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    # Find y-range
    all_vals = []
    for _, values in series:
        all_vals.extend(values)

    y_min = min(all_vals)
    y_max = max(all_vals)

    # Add padding
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.02
    y_min = max(0.0, y_min - pad)
    y_max = min(1.0, y_max + pad)   # assumes accuracy is in [0, 1]

    def x_to_px(epoch: int) -> float:
        return left + (epoch - 1) / (n_epochs - 1) * plot_width

    def y_to_px(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    # Axes
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)

    # Title and axis labels
    draw.text((width // 2 - 140, 20), title, fill="black", font=font)
    draw.text((width // 2 - 20, height - 30), "Epoch", fill="black", font=font)
    draw.text((15, height // 2), "Accuracy", fill="black", font=font)

    # Y ticks
    n_y_ticks = 5
    for i in range(n_y_ticks + 1):
        frac = i / n_y_ticks
        y_val = y_min + frac * (y_max - y_min)
        y = y_to_px(y_val)

        draw.line((left, y, width - right, y), fill=(220, 220, 220), width=1)
        draw.text((30, y - 7), f"{y_val:.2f}", fill="black", font=font)

    # X ticks
    for epoch in range(1, n_epochs + 1):
        x = x_to_px(epoch)
        draw.line((x, height - bottom, x, height - bottom + 5), fill="black", width=1)

        # only label some ticks if there are many epochs
        if n_epochs <= 20 or epoch == 1 or epoch == n_epochs or epoch % 5 == 0:
            draw.text((x - 8, height - bottom + 10), str(epoch), fill="black", font=font)

    # Fixed colours for up to 4 lines
    colours = [
        (40, 90, 200),    # blue
        (220, 70, 60),    # red
        (50, 160, 80),    # green
        (180, 120, 40),   # brown/orange
    ]

    # Draw lines
    for idx, (label, values) in enumerate(series):
        colour = colours[idx % len(colours)]
        points = [(x_to_px(i + 1), y_to_px(v)) for i, v in enumerate(values)]

        if len(points) >= 2:
            draw.line(points, fill=colour, width=3)

    # Legend
    legend_x = width - 200
    legend_y = top + 10
    legend_height = 25 * len(series) + 15
    draw.rectangle(
        (legend_x, legend_y, legend_x + 180, legend_y + legend_height),
        outline="black",
        fill="white"
    )

    for idx, (label, _) in enumerate(series):
        colour = colours[idx % len(colours)]
        y = legend_y + 15 + idx * 25
        draw.line((legend_x + 10, y, legend_x + 35, y), fill=colour, width=3)
        draw.text((legend_x + 45, y - 7), label, fill="black", font=font)

    img.save(save_path)
    print(f"Saved plot to {save_path}")

def config_cuda() -> Tuple[torch.device, bool]:
    """
    Helper function which checks if CUDA is available and returns the appropriate device and a boolean flag for use_cuda,
    which can be used for pin_memory in dataloaders and non_blocking transfers to GPU
    
    inputs: None

    outputs: device (torch.device),
             use_cuda (bool)
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device, use_cuda

# ----------------------------- NETWORK -----------------------------

class Net(nn.Module):
    def __init__(self, drop_prob:float = 0.1, drop_block_size:int=3):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.dropblock1 = DropBlock2d(p=drop_prob, block_size=drop_block_size) # dropblock
        self.dropblock2 = DropBlock2d(p=drop_prob, block_size=drop_block_size)

        self.pool = nn.MaxPool2d(2, 2) #using only maxpooling instead of say average pooling, as it is more common in image classification tasks
        # seems like it would be better to pick up on sharp edges or other strong features

        #output classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.bn2(self.conv4(x))) 
        x = self.dropblock1(x)
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.dropblock2(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------ TRAIN ------------------------------

def make_dataloaders(batch_size: int, use_cuda: bool) -> Tuple[DataLoader, DataLoader]:
    """
    Setup dataloaders for training and validation.
    This function loads the CIFAR-100 training dataset, applies transformations, splits it into training and validation sets,

    inputs: batch_size (int)
            use_cuda (bool)

    outputs: train_loader (DataLoader)
             val_loader (DataLoader)
             n_classes (int)
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    full_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # generator = torch.Generator().manual_seed(42) Can use a generator if i want to keep the same split across runs, but not necessary for this task

    trainset, valset = random_split(full_dataset, [train_size, val_size]) # generator=generator)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=use_cuda)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=use_cuda)
    n_classes = len(full_dataset.classes)

    return train_loader, val_loader, n_classes

def train_epoch(net: Net, 
                train_loader: DataLoader, 
                criterion: torch.nn.CrossEntropyLoss, 
                optimizer: optim.SGD, 
                device: torch.device, 
                use_cuda: bool) -> float:
    """
    trains the model for one epoch, iterating over the training dataloader, and returns the average training loss for that epoch
    
    inputs: net (Net)
            train_loader (DataLoader)
            criterion (torch.nn.CrossEntropyLoss)
            optimizer (optim.SGD)
            device (torch.device)
            use_cuda (bool)

    outputs: accuracy (float)
    """
    
    net.train()

    # setup up correct and total count to calculate accuracy
    correct = 0
    total = 0
    for data in train_loader: # train_loader is an iterable itself
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda) # for GPU
        
        # zero the parameter gradients
        optimizer.zero_grad(set_to_none=True) 

        # forward + backward + optimize
        outputs = net(inputs)
        predictions = outputs.argmax(dim=1)

        # criterion itself performs label smoothing if provided as an argument to the instance
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total

    return accuracy
        
def validate_epoch(net: Net, 
                   val_loader: DataLoader, 
                   device: torch.device, 
                   use_cuda: bool) -> float:
    """
    validates the model for one epoch, iterating over the validation dataloader, and returns the average validation loss for that epoch

    inputs: net (Net)
            val_loader (DataLoader)
            device (torch.device)
            use_cuda (bool)
    
    outputs: accuracy (float)
    """
    # Validation no training so we don't need to track gradients  
    # 
    net.eval()

    # setup up correct and total count to calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad(): # disable gradient tracking for validation  
        for data in val_loader: # val_loader is also an iterable 
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda) # for GPU

            outputs = net(inputs)
            predictions = outputs.argmax(dim=1)


            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy

def train_model(epochs: int, 
                patience: int, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                net: Net, 
                criterion: torch.nn.CrossEntropyLoss, 
                optimizer: optim.Adam, 
                regularise_flag: str,
                device: torch.device, 
                use_cuda: bool) -> Tuple[list, list]:
    """
    trains the model for a specified number of epochs, while also performing validation at the end of each epoch
    implements early stopping based on validation loss with a specified patience. 
    It also tracks training and validation loss over epochs for plotting later.

    inputs: epochs (int)
            patience (int)
            train_loader (DataLoader)
            val_loader (DataLoader)
            net (Net)
            criterion (torch.nn.CrossEntropyLoss)
            optimizer (optim.SGD)
            device (torch.device)
            use_cuda (bool)

    outputs: train_loss (list of float)
             val_loss (list of float)
    """
    # setup early stopping tracking variables
    best_val_accuracy = float(0.0)
    epochs_no_improve = 0
    min_delta = 1e-4

    # setup lists to track training and validation loss over epochs for plotting later
    train_accuracy = []
    val_accuracy = []
    # train
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}/{epochs}")

        training_accuracy = train_epoch(net, train_loader, criterion, optimizer, device, use_cuda)
        train_accuracy.append(training_accuracy)

        validate_accuracy = validate_epoch(net, val_loader, device, use_cuda)
        val_accuracy.append(validate_accuracy)

        print(f"Training accuracy: {training_accuracy:.4f} | Validation accuracy: {validate_accuracy:.4f}")

        if val_accuracy[-1] > best_val_accuracy - min_delta:
            best_val_accuracy = val_accuracy[-1]
            epochs_no_improve = 0
            #torch.save(net.state_dict(), f"best_model_{regularise_flag}_task1.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break   

    return train_accuracy, val_accuracy

def grid_search(
    device: torch.device,
    use_cuda: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 40,   # shorter for search
    patience: int = 3 # shorter for search
):
    """
    Performs a grid search over DropBlock and weight decay hyperparameters.

    inputs:
        device (torch.device) which device to use
        use_cuda (bool) true or false is cuda is being used
        train_loader (DataLoader) dataset for training data
        val_loader (DataLoader) dataset for validation data
        epochs (int)
        patience (int) number of runs to wait with no increase in validation accuracy before breaking early
        
    Returns:
        best_config (dict)
        all_results (list of dict)
    """

    learning_rate = [0.05, 0.01]
    momentum = [0.9, 0.85, 0.95]
    drop_probs = [0.05, 0.1, 0.15]
    block_sizes = [1, 3, 5]
    weight_decays = [1e-4, 5e-4, 1e-3]

    results = []
    total_runs = len(drop_probs) * len(block_sizes) * len(weight_decays) * len(learning_rate) * len(momentum)
    run_idx = 0

    for learning_rate, momentum, drop_prob, block_size, w_decay in product(learning_rate, momentum, drop_probs, block_sizes, weight_decays):
        run_idx += 1

        print("\n" + "="*50)
        print(f"Run {run_idx}/{total_runs}")
        print(f"learning_rate={learning_rate}, momentum={momentum}, drop_prob={drop_prob}, block_size={block_size}, weight_decay={w_decay}")

        # fresh model
        net = Net(drop_prob=drop_prob, drop_block_size=block_size).to(device)

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum, weight_decay=w_decay)

        # train
        train_accuracy, val_accuracy = train_model(epochs,patience,train_loader,val_loader,net,criterion,optimizer,"grid_search",device,use_cuda)

        best_val = max(val_accuracy)
        final_val = val_accuracy[-1]
        gap = calculate_mean_gap(train_accuracy, val_accuracy)

        result = {"drop_prob": drop_prob,"block_size": block_size,"weight_decay": w_decay,"learning_rate": learning_rate,"momentum": momentum,"best_val_accuracy": best_val,"final_val_accuracy": final_val,"mean_gap": gap}

        results.append(result)

        print(f"Best Val Loss: {best_val:.4f}")
        print(f"Final Val Loss: {final_val:.4f}")
        print(f"Mean Gap: {gap:.4f}")

    # sort by best validation loss
    results.sort(key=lambda x: x["best_val_accuracy"], reverse=True)

    print("\n" + "="*50)
    print("TOP 5 CONFIGS:")
    for r in results[:5]:
        print(r)

    with open("results.txt", "w") as f:

        f.write("\nTOP 5 CONFIGS (sorted by best_val_accuracy):\n")
        f.write("="*50 + "\n")

        for r in results[:5]:
            f.write(str(r) + "\n")

        f.write("\n" + "="*50 + "\n")
        f.write("ALL RESULTS (sorted by best_val_accuracy)\n")
        f.write("="*50 + "\n")

        for r in results:
            f.write(str(r) + "\n")

    best_config = results[0]
    return best_config, results

if __name__ == '__main__':
    """
    main execution block, sets up hyperparameters, dataloaders, model, and starts training, while also plotting the accuracy curves at the end
    """

    batch_size = 128
    epochs = 100 
    patience = 3 # for early stopping, if validation loss does not improve for this many epochs, stop training

    learning_rate = 0.05 # for SGD
    momentum = 0.9 # for SGD

    run_grid_search = True

    device, use_cuda = config_cuda()
    train_loader, val_loader, n_classes = make_dataloaders(batch_size, use_cuda)
    
    if run_grid_search:
        best_config, all_results = grid_search(device,use_cuda,train_loader,val_loader)
        print("\nBest configuration found:")
        print(best_config)
        exit()

    # ------- Train unregularised model -------
    # hyperparams
    regularise_flag = "no_regularisation"
    print(f"Training {regularise_flag} model")

    # init net with dropBlock
    net = Net(drop_prob=0, drop_block_size=1).to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum)
    
    train_accuracy_unreg, val_accuracy_unreg = train_model(epochs, patience, train_loader, val_loader, net, criterion, optimizer, regularise_flag, device, use_cuda)

    print('Training done.')

    mean_gap = calculate_mean_gap(train_accuracy_unreg, val_accuracy_unreg)
    print(f"Mean gap between training and validation accuracy across epochs for {regularise_flag}: {mean_gap:.4f}")
    # save trained model
    print(f"Best model saved to best_model_{regularise_flag}_task1.pt")


    # ------- Train regularised model -------

    # hyperparams
    w_decay = 5e-4 # best params
    drop_prob = 0.05 # best params
    drop_block_size = 3 # best params
    regularise_flag = "regularisation"

    print(f"Training {regularise_flag} model")

    net = Net(drop_prob=drop_prob, drop_block_size=drop_block_size).to(device)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum,weight_decay=w_decay)   
    
    train_accuracy_reg, val_accuracy_reg = train_model(epochs, patience, train_loader, val_loader, net, criterion, optimizer, regularise_flag, device, use_cuda)

    print('Training done.')

    mean_gap = calculate_mean_gap(train_accuracy_reg, val_accuracy_reg)
    print(f"Mean gap between training and validation accuracy across epochs for {regularise_flag}: {mean_gap:.4f}")
    # save trained model
    print(f"Best model saved to best_model_{regularise_flag}_task1.pt")


    #lines = [("unregularised train accuracy", train_accuracy_unreg),("unregularised val accuracy", val_accuracy_unreg),("regularised train accuracy", train_accuracy_reg),("regularised val accuracy", val_accuracy_reg)]
    #print(lines)
    #draw_accuracy_comparison_plot(lines, save_path="generalization_gap.png", title="Training vs Validation Accuracy Regularised vs Unregularised")
