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

from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

# sanity checks to make sure my GPU was being used
#print(torch.cuda.is_available())
#print(torch.cuda.device_count())
#print(torch.cuda.current_device())
#print(torch.cuda.device(0))
#print(torch.cuda.get_device_name(0))  



# we can't have util files annoyingly, or I would try organise it a bit more...

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
    draw.text((15, (height/2)), "Loss", fill="black", font=font)

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

# ----------------------------- NETWORK -----------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

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
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------------------ TRAIN ------------------------------

def to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert labels to one-hot encoded tensors
    inputs:
        labels (torch.Tensor): A tensor of shape (batch_size,) containing class indices
        num_classes (int): The total number of classes for one-hot encoding 

    outputs:
        one_hot_labels (torch.Tensor): A tensor of shape (batch_size, num_classes) containing one-hot encoded labels
    """

    if len(labels.shape) != 1:
        raise ValueError("labels must be a 1D tensor of shape (batch_size,)")

    one_hot_labels = torch.zeros(labels.size(0), num_classes, device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)
    
    return one_hot_labels

class soft_cross_entropy_loss(torch.nn.Module):
    """
    Soft cross entropy loss implementation
    """

    def __init__(self, num_classes: int = None, smoothing: float = 0):
        super(soft_cross_entropy_loss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

        if smoothing < 0.0 or smoothing > 1.0:
            raise ValueError("smoothing must be in the range [0, 1]")

    def softmax(self, logits:torch.Tensor) -> torch.Tensor:
        """
        softmax implementation, deprecated in favour of log_softmax for numerical stability, 
        but left here for completeness and potential use in extra credit

        inputs: logits (torch.Tensor)

        outputs: softmaxed_x (torch.Tensor)
        """

        # subtract max for numerical stability
        x_max = torch.max(logits, dim=-1, keepdim=True).values
        e_x = torch.exp(logits - x_max)
        softmaxed_x = e_x / e_x.sum(dim=-1, keepdim=True)
        
        return softmaxed_x

    def label_smoothing(self, labels: torch.Tensor) -> torch.Tensor:
        """
        label_smoothing implementation, which smooths the one-hot encoded labels by distributing some of the probability mass to the other classes

        inputs: 
            labels (torch.Tensor) - expected to be one-hot encoded

        outputs: 
            smoothed_labels (torch.Tensor)
        """
        
        if self.smoothing == 0.0:
            return labels
        
        if labels.shape[-1] != self.num_classes:
            raise ValueError("labels shape does not match num_classes for label smoothing")

        smoothed_labels = labels * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        return smoothed_labels

    def log_softmax(self, logits:torch.Tensor) -> torch.Tensor:
        """
        log_softmax omputes the log softmax of a vector or batch of vectors
        this is more numerically stable than taking the log of the softmax, and is what is used in the cross entropy loss implementation in PyTorch

        inputs: logits (torch.Tensor)

        outputs: log_softmaxed_x (torch.Tensor)
        """

        # taken from https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
        # used both max shifting and the log-sum-exp trick for numerical stability, but this can still underflow for very small values
        # subtract max for numerical stability

        """
        c = x.max()
        logsumexp = np.log(np.exp(x - c).sum())
        return x - c - logsumexp
        """
        # shifted_logits is identical to (x-c)
        shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
        log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True))
        return shifted_logits - log_sum_exp

    def soft_cross_entropy(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        soft_cross_entropy is a helper function to compute the cross entropy loss given the model outputs and true labels
        !!!this is HARDCODED for a classification, i.e. expects outputs to be of shape (batch_size, num_classes)!!!

        inputs: outputs (torch.Tensor)
                labels (torch.Tensor)

        outputs: loss (torch.Tensor)
        
        cross entropy loss is:
        
        -1/i * sum_i[ sum_j[ t_ij * log(p_ij) ] ]

        where i is the number of batches, j the number of classes, t_ij is the true label (one-hot encoded)
        and p_ij is the predicted probability for class j in batch i

        p_ij is given by the softmax of the outputs, but we can compute the log of the softmax more stably
        using the log_softmax function defined above, which combines the softmax and log in a more numerically
        stable way
        
        Then we just need to multiply the log softmax outputs by the one-hot encoded labels and take the mean
        over batches to get the loss
        """
        # convert to one-hot
        if len(labels.shape) == 1:
            if self.num_classes:
                labels = to_one_hot(labels, self.num_classes)
            else:
                if labels.max() <= outputs.shape[1]:
                    labels = to_one_hot(labels, outputs.shape[1]) # if num_classes not specified, infer from outputs shape
                else:
                    raise ValueError("num_classes not specified and labels contain values greater than outputs shape, cannot infer num_classes")
        elif labels.shape != outputs.shape:
            raise ValueError("labels must be either a 1D tensor of shape (batch_size,) or a one-hot encoded tensor of shape (batch_size, num_classes)")

        labels = self.label_smoothing(labels) # if no smoothing, this will just return the original one-hot labels

        log_probs = self.log_softmax(outputs)

        loss_per_sample = -(labels * log_probs).sum(dim=-1) # tensor operations effectively do the sum over classes
                                                            # for each sample in the batch, giving us a tensor of shape (batch_size,) 
                                                            # need to be careful to do dim -1 so it applies over the classes and not the batch dimension

        return loss_per_sample.mean()

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        forward function to be consistent with PyTorch loss function interface, simply calls the soft_cross_entropy helper function defined above

        inputs: outputs (torch.Tensor)
                labels (torch.Tensor)
        
        outputs: loss (torch.Tensor)
        """
        return self.soft_cross_entropy(outputs, labels)

class MixUp():
    """
    MixUp data augmentation technique for training neural networks
    MixUp creates new training samples by taking convex combinations of pairs of examples and their labels.
    
    """
    def __init__(self, alpha: float = 1.0, num_classes: int = None, active: bool = True):
        """
        Initialize the MixUp data augmentation technique.

        inputs:
            alpha (float): The alpha parameter for the Beta distribution used to sample the mixing coefficient.
            num_classes (int): The number of classes in the classification task, needed for one-hot encoding of labels when applying MixUp to labels.
            active (bool): Whether to apply MixUp augmentation when the instance is called. If False, the instance will return the original inputs and labels without mixing.
            
        outputs:
            Mixup object that can be called to apply MixUp augmentation to a batch of inputs and labels
        """

        if alpha <= 0:
            raise ValueError("alpha must be greater than 0 for MixUp")
        
        self.active = active
        self.alpha = alpha
        self.num_classes = num_classes

    def sample_beta(self) -> torch.Tensor:
        """
        sample_beta samples a mixing coefficient lambda from a Beta distribution parameterized by alpha, as recommended in the original MixUp paper
        use Gamma sampling, since Beta can be sampled by sampling two Gamma distributions and taking the ratio
        
        Finding more primitive methods to replace torch.distributions.Beta or .Gamma become harder, and need specialised algorithms
        or rejection sampling.

        inputs: None

        outputs: lam (torch.Tensor) - a mixing coefficient sampled from the Beta distribution, constrained to be in [0.5, 1] to avoid too much mixing

        """
        gamma1 = torch.distributions.Gamma(self.alpha, 1.0).sample()
        gamma2 = torch.distributions.Gamma(self.alpha, 1.0).sample()

        lam = gamma1 / (gamma1 + gamma2)

        # ensure lambda is in [0.5, 1] to avoid too much mixing, as recommended in the original MixUp paper
        # specifically, each image is seen at least once with a weight of 0.5 or more, to preserve some of the original
        # image information and prevent underfitting
        return max(lam, 1 - lam) 

    def __call__(self, inputs: torch.Tensor, labels: torch.Tensor, *, lambda_param: float=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp augmentation to a batch of inputs and labels.

        inputs:
            inputs (torch.Tensor): A batch of input data, expected to be of shape (batch_size, channels, height, width)
            labels (torch.Tensor): A batch of labels, expected to be of shape (batch_size,)
            lambda_param (float, optional): A mixing coefficient to use for this batch. If None, a random lambda will be sampled from the Beta distribution. (keyword only)

        outputs:
            inputs (torch.Tensor): A batch of mixed input data, expected to be of shape (batch_size, channels, height, width)
            labels (torch.Tensor): A batch of mixed labels, expected to be of shape (batch_size, num_classes)
        """
        if not self.active:
            return inputs, labels

        batch_size = inputs.size(0)

        if batch_size < 2:
            # cannot apply MixUp to a batch of size 1, so just return the original inputs and labels
            return inputs, labels

        # either the last dimension shows each label has size 1, or the labels are a 1D tensor of shape 
        # (batch_size,), in either case we need to convert to one-hot encoding for mixing
        if labels.shape[-1] == 1 or (len(labels.shape) == 1 and labels.shape[0] == batch_size): # if labels are not one-hot encoded, we need to convert them to one-hot encoding before mixing
            if self.num_classes:
                labels = to_one_hot(labels, self.num_classes)
            else:
                raise ValueError("num_classes must be specified for MixUp when labels are not already one-hot encoded")

        if lambda_param is None:
            lambda_param = self.sample_beta() # sample a lambda for each example in the batch

        permutation = torch.randperm(batch_size) # generate a random permutation of the batch indices to mix


        mixed_inputs = lambda_param * inputs + (1 - lambda_param) * inputs[permutation] # mix the inputs according to the permutation and lambda
        mixed_labels = lambda_param * labels + (1 - lambda_param) * labels[permutation] # mix the labels according to the same permutation and lambda

        return mixed_inputs, mixed_labels

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
                criterion: soft_cross_entropy_loss, 
                optimizer: optim.SGD, 
                mixup: MixUp,
                device: torch.device, 
                use_cuda: bool) -> float:
    """
    trains the model for one epoch, iterating over the training dataloader, and returns the average training loss for that epoch
    
    inputs: net (Net)
            train_loader (DataLoader)
            criterion (torch.nn.CrossEntropyLoss)
            optimizer (optim.SGD)
            mixup (MixUp)
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
        inputs, labels = mixup(inputs=inputs, labels=labels) # apply MixUp augmentation to the inputs and labels
        
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
                   criterion: soft_cross_entropy_loss, 
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
                criterion: soft_cross_entropy_loss, 
                optimizer: optim.SGD, 
                mixup: MixUp,
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
            mixup (MixUp)
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

        training_loss = train_epoch(net, train_loader, criterion, optimizer, mixup, device, use_cuda)
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

    smoothing_factor = 0.1 # for label smoothing
    alpha = 0.4 # for mixup

    device, use_cuda = config_cuda()
    train_loader, val_loader, n_classes = make_dataloaders(batch_size, use_cuda)
    
    net = Net().to(device)

    active = True
    # custom mixup implementation, with specified alpha and number of classes for one-hot encoding
    mixup = MixUp(alpha=alpha, num_classes=n_classes, active=active) # example of how to initialize MixUp, not currently used in training loop but can be easily integrated by applying it to the inputs and labels in the train_epoch function before the forward pass
    ## loss and optimiser
    # Custom soft_cross_entropy_loss function, also implements label smoothing with given smoothing factor
    criterion = soft_cross_entropy_loss(n_classes, smoothing_factor)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)    
    
    train_loss, val_loss = train_model(epochs, patience, train_loader, val_loader, net, criterion, optimizer, mixup, device, use_cuda)

    print('Training done.')

    mixup_str = f"MixUp_{alpha}" if active else "NoMixUp"
    smoothing_str = f"Smoothing_{smoothing_factor}" if smoothing_factor > 0 else "NoSmoothing"
    filename = f"loss_plot_{mixup_str}_{smoothing_str}.png"
    
    mixup_str = f"MixUp (alpha={str(alpha).replace(".", "_")})" if active else "No MixUp"
    smoothing_str = f"Label Smoothing (factor={str(smoothing_factor).replace(".", "_")})" if smoothing_factor > 0 else "No Smoothing"
    title = f"Training vs Validation Loss ({mixup_str}, {smoothing_str})"

    draw_loss_plot(train_loss, val_loss, save_path=filename, title=title)
    mean_gap = calculate_mean_gap(train_loss, val_loss)
    print(f"Mean gap between training and validation loss across epochs: {mean_gap:.4f}")
    # save trained model
    print("Best model saved to best_model_task2.pt")

    