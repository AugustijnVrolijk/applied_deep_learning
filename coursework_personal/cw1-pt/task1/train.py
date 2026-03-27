# training script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from torch.utils.data import random_split, DataLoader
from PIL import Image, ImageDraw, ImageFont

from network import Net

from typing import Tuple

#print(torch.cuda.is_available()) sanity checks it was running on my GPU
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))


def draw_accuracy_plot(
    train_acc,
    val_acc,
    save_path="accuracy_plot.png",
    width=1000,
    height=700,
    title="Training and Validation Accuracy"):
    """
    draw_accuracy_plot creates a line plot of training and validation accuracy over epochs and saves it as an image using Pillow
    
    This function was primarily created with AI assistance, and is not the main focus of the assignment
    """

    if len(train_acc) != len(val_acc):
        raise ValueError("train_acc and val_acc must have the same length")

    if len(train_acc) < 2:
        raise ValueError("Need at least 2 epochs to draw a line plot")

    # Create blank canvas
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Try default font
    font = ImageFont.load_default()

    # Margins
    left = 100
    right = 50
    top = 70
    bottom = 100

    plot_width = width - left - right
    plot_height = height - top - bottom

    # Data range
    n_epochs = len(train_acc)
    x_min, x_max = 1, n_epochs

    # Accuracy usually in [0,1], but adapt to data with a bit of padding
    all_vals = train_acc + val_acc
    y_min_data = min(all_vals)
    y_max_data = max(all_vals)

    # Nice fixed range if values are probabilities
    y_min = min(0.0, y_min_data - 0.02)
    y_max = max(1.0, y_max_data + 0.02)

    if y_max == y_min:
        y_max = y_min + 1e-6

    def x_to_px(epoch):
        # epoch is 1-indexed
        return left + (epoch - x_min) / (x_max - x_min) * plot_width

    def y_to_px(acc):
        # invert y because image coordinates grow downward
        return top + (y_max - acc) / (y_max - y_min) * plot_height

    # Draw axes
    axis_color = "black"
    draw.line((left, top, left, height - bottom), fill=axis_color, width=2)                # y-axis
    draw.line((left, height - bottom, width - right, height - bottom), fill=axis_color, width=2)  # x-axis

    # Title
    draw.text((width // 2 - 120, 20), title, fill="black", font=font)

    # Axis labels
    draw.text((width // 2 - 20, height - 40), "Epoch", fill="black", font=font)
    draw.text((20, top - 20), "Accuracy", fill="black", font=font)

    # Grid + y ticks
    y_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    grid_color = (220, 220, 220)

    for yv in y_ticks:
        if y_min <= yv <= y_max:
            y = y_to_px(yv)
            draw.line((left, y, width - right, y), fill=grid_color, width=1)
            draw.text((50, y - 7), f"{yv:.1f}", fill="black", font=font)

    # X ticks
    for epoch in range(1, n_epochs + 1):
        x = x_to_px(epoch)
        draw.line((x, height - bottom, x, height - bottom + 5), fill="black", width=1)
        draw.text((x - 5, height - bottom + 10), str(epoch), fill="black", font=font)

    # Convert series to points
    train_points = [(x_to_px(i + 1), y_to_px(v)) for i, v in enumerate(train_acc)]
    val_points = [(x_to_px(i + 1), y_to_px(v)) for i, v in enumerate(val_acc)]

    # Draw lines
    train_color = (40, 90, 200)   # blue-ish
    val_color = (220, 70, 60)     # red-ish

    draw.line(train_points, fill=train_color, width=3)
    draw.line(val_points, fill=val_color, width=3)

    # Draw markers
    r = 4
    for x, y in train_points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=train_color)

    for x, y in val_points:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=val_color)

    # Legend
    legend_x = width - 220
    legend_y = top + 10

    draw.rectangle((legend_x, legend_y, legend_x + 160, legend_y + 60), outline="black", fill="white")
    draw.line((legend_x + 10, legend_y + 18, legend_x + 40, legend_y + 18), fill=train_color, width=3)
    draw.text((legend_x + 50, legend_y + 10), "Train accuracy", fill="black", font=font)

    draw.line((legend_x + 10, legend_y + 43, legend_x + 40, legend_y + 43), fill=val_color, width=3)
    draw.text((legend_x + 50, legend_y + 35), "Val accuracy", fill="black", font=font)

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

    return train_loader, val_loader

def setup_model(device: torch.device) -> Tuple[Net, torch.nn.CrossEntropyLoss, optim.SGD]:
    """
    loads the neural network, loss function, and optimizer.
    
    inputs: device (torch.device)

    outputs: net (Net) 
             criterion (torch.nn.CrossEntropyLoss)
             optimizer (optim.SGD)
    """
    net = Net().to(device)
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    return net, criterion, optimizer

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

        if val_loss[-1] < best_val_loss - min_delta:
            best_val_loss = val_loss[-1]
            epochs_no_improve = 0
            torch.save(net.state_dict(), "best_model.pt")
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
    patience = 3

    device, use_cuda = config_cuda()
    train_loader, val_loader = make_dataloaders(batch_size, use_cuda)
    net, criterion, optimizer = setup_model(device)
    train_loss, val_loss = train_model(epochs, patience, train_loader, val_loader, net, criterion, optimizer, device, use_cuda)

    print('Training done.')

    draw_accuracy_plot(train_loss, val_loss, save_path="accuracy_plot.png")
    # save trained model
    print("Best model saved to best_model.pt")

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