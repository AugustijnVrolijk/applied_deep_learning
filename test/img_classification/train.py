# train script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from network import Net

from PIL import Image, ImageDraw, ImageFont

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

if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 20
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # example images
    dataiter = iter(trainloader)
    images, labels = next(dataiter) # note: for pytorch versions (<1.14) use dataiter.next()

    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("train_pt_images.jpg")
    print('train_pt_images.jpg saved.')
    print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))


    ## cnn
    net = Net()


    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_loss = []
    ## train
    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches
                train_loss.append(running_loss / 500)
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    print('Training done.')
    val_acc   = [0 for i in range(len(train_loss))]

    draw_accuracy_plot(train_loss, val_acc, save_path="accuracy_plot.png")
    # save trained model
    torch.save(net.state_dict(), 'saved_model.pt')
    print('Model saved.')