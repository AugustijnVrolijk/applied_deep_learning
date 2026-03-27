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
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import random_split, DataLoader
from PIL import Image, ImageDraw, ImageFont

from network import Net
from typing import Tuple

# sanity checks to make sure my GPU was being used
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))  

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

def draw_loss_plot(train_loss: list[float],val_loss: list[float],save_path: str = "loss_plot.png",title: str = "Training vs Validation Loss") -> None:
    """
    Draw a simple line plot of training and validation loss over epochs using Pillow.
    This function was generated using gen AI and is only used for visualisation support for the writeup

    inputs:
        train_loss: list of training losses for each epoch
        val_loss: list of validation losses for each epoch
        save_path: output image path
        width: image width in pixels
        title: plot title

    outputs:
        None (saves the image to save_path)
    """

    width = 900
    height = 600

    if len(train_loss) != len(val_loss):
        raise ValueError("train_loss and val_loss must have the same length")

    if len(train_loss) < 2:
        raise ValueError("Need at least 2 epochs to draw a line plot")

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Margins
    left = 80
    right = 40
    top = 50
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom

    n_epochs = len(train_loss)

    all_vals = train_loss + val_loss
    y_min = min(all_vals)
    y_max = max(all_vals)

    # add small padding so lines do not touch borders
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    y_min -= pad
    y_max += pad

    def x_to_px(epoch: int) -> float:
        return left + (epoch - 1) / (n_epochs - 1) * plot_width

    def y_to_px(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    # Axes
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)

    # Title and labels
    draw.text((width // 2 - 80, 15), title, fill="black", font=font)
    draw.text((width // 2 - 20, height - 30), "Epoch", fill="black", font=font)
    draw.text((15, (top/2)), "Loss", fill="black", font=font)

    # Y ticks
    n_y_ticks = 5
    for i in range(n_y_ticks + 1):
        frac = i / n_y_ticks
        y_val = y_min + frac * (y_max - y_min)
        y = y_to_px(y_val)

        draw.line((left, y, width - right, y), fill=(220, 220, 220), width=1)
        draw.text((45, y - 7), f"{y_val:.2f}", fill="black", font=font)

    # X ticks
    for epoch in range(1, n_epochs + 1):
        x = x_to_px(epoch)
        draw.line((x, height - bottom, x, height - bottom + 5), fill="black", width=1)
        draw.text((x - 5, height - bottom + 10), str(epoch), fill="black", font=font)

    train_points = [(x_to_px(i + 1), y_to_px(v)) for i, v in enumerate(train_loss)]
    val_points = [(x_to_px(i + 1), y_to_px(v)) for i, v in enumerate(val_loss)]

    # Colours
    train_color = (40, 90, 200)
    val_color = (220, 70, 60)

    # Draw lines
    draw.line(train_points, fill=train_color, width=3)
    draw.line(val_points, fill=val_color, width=3)

    # Legend
    legend_x = width - 180
    legend_y = top + 10
    draw.rectangle((legend_x, legend_y, legend_x + 140, legend_y + 50), outline="black", fill="white")

    draw.line((legend_x + 10, legend_y + 15, legend_x + 35, legend_y + 15), fill=train_color, width=3)
    draw.text((legend_x + 45, legend_y + 8), "Train loss", fill="black", font=font)

    draw.line((legend_x + 10, legend_y + 35, legend_x + 35, legend_y + 35), fill=val_color, width=3)
    draw.text((legend_x + 45, legend_y + 28), "Val loss", fill="black", font=font)

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

    outputs: train_loss_normal (float)
    """
    
    # setup loss tracking, and iter count to normalise at the end
    net.train()
    training_loss = 0.0
    n_iter = 0
    for data in train_loader: # train_loader is an iterable itself
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda) # for GPU
        
        # zero the parameter gradients
        optimizer.zero_grad(set_to_none=True) 

        # forward + backward + optimize
        outputs = net(inputs)
        # criterion itself performs label smoothing if provided as an argument to the instance
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
        n_iter += 1

    train_loss_normal = training_loss / n_iter
    return train_loss_normal
        
def validate_epoch(net: Net, 
                   val_loader: DataLoader, 
                   criterion: torch.nn.CrossEntropyLoss, 
                   device: torch.device, 
                   use_cuda: bool) -> float:
    """
    validates the model for one epoch, iterating over the validation dataloader, and returns the average validation loss for that epoch

    inputs: net (Net)
            val_loader (DataLoader)
            criterion (torch.nn.CrossEntropyLoss)
            device (torch.device)
            use_cuda (bool)
    
    outputs: validation_loss_normal (float)
    """
    # Validation no training so we don't need to track gradients  
    # 
    # setup loss tracking, and iter count to normalise at the end
    net.eval()
    validation_loss = 0.0  
    n_iter = 0
    with torch.no_grad(): # disable gradient tracking for validation  
        for data in val_loader: # val_loader is also an iterable 
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=use_cuda), labels.to(device, non_blocking=use_cuda) # for GPU

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()
            n_iter += 1

    validation_loss_normal = validation_loss / n_iter
    return validation_loss_normal

def train_model(epochs: int, 
                patience: int, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                net: Net, 
                criterion: torch.nn.CrossEntropyLoss, 
                optimizer: optim.SGD, 
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
    best_val_loss = float('inf')
    epochs_no_improve = 0
    min_delta = 1e-4

    # setup lists to track training and validation loss over epochs for plotting later
    train_loss = []
    val_loss = []
    # train
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch+1}/{epochs}")

        training_loss = train_epoch(net, train_loader, criterion, optimizer, device, use_cuda)
        train_loss.append(training_loss)

        validate_loss = validate_epoch(net, val_loader, criterion, device, use_cuda)
        val_loss.append(validate_loss)

        print(f"Training Loss: {training_loss:.4f} | Validation Loss: {validate_loss:.4f}")

        if val_loss[-1] < best_val_loss - min_delta:
            best_val_loss = val_loss[-1]
            epochs_no_improve = 0
            torch.save(net.state_dict(), "best_model_task2.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break   

    return train_loss, val_loss

if __name__ == '__main__':
    """
    main execution block, sets up hyperparameters, dataloaders, model, and starts training, while also plotting the accuracy curves at the end
    """

    batch_size = 128
    epochs = 100 
    patience = 5 # for early stopping, if validation loss does not improve for this many epochs, stop training

    learning_rate = 0.005 # for SGD
    momentum = 0.9 # for SGD

    device, use_cuda = config_cuda()
    train_loader, val_loader, n_classes = make_dataloaders(batch_size, use_cuda)
    
    net = Net().to(device)


    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)    
    
    train_loss, val_loss = train_model(epochs, patience, train_loader, val_loader, net, criterion, optimizer, mixup, device, use_cuda)

    print('Training done.')

    draw_loss_plot(train_loss, val_loss, save_path="loss_plot_no_MixUp.png", title="Training vs Validation Loss without MixUp or Label Smoothing")
    mean_gap = calculate_mean_gap(train_loss, val_loss)
    print(f"Mean gap between training and validation loss across epochs: {mean_gap:.4f}")
    # save trained model
    print("Best model saved to best_model_task2.pt")

    

    """
    DO WEIGHT DECAY
    ALSO ADD DROPOUT? (DropBlock?)

    AND ALSO DO RANDOM SHIFT AUGMENTATION (translate the images around a little bit) (small dataset)

    GEN AI STATEMENT, DISCUSS HOW IT INITALLY SUGGESTED A FAR TOO BIG NETWORK (many many layers, and final channel size of like 1024)
    MASSIVELY OVERFITTING, AND EVALUATION SCORE STAYED REALLY LOW: ~15%
    DUE TO THIS I MANUALLY EDITED THE SIZE A LOT AND PLAYED AROUND WITH OUTPUT SIZES AND LAYER NUMBERS AND ADDING MAXPOOL ETC...
    THIS HAD A HUGE EFFECT; SO FUTURE IMPLEMENTATION COULD LACK AT IMPLEMENTING ARCHITECHURE SEARCH TO AUTOMATE THIS PROCESS, AS IT IS VERY TIME CONSUMING,
    COULD PROBABLY BE DONE BETTER ALGORITHMICALLY (Genetic Algorithms optimisation over architecture hyperparameters) THIS SEEMED
    TO HAVE THE LARGEST EFFECT ON PERFORMANCE, MORE SO THAN OPTIMISING OTHER HYPERPARAMETERS LIKE LEARNING RATE, BATCH SIZE, OR EPOCHS, OR OTHER REGULARISATION
    , WHICH HAD MORE MARGINAL EFFECTS
    """