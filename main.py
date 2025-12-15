import random, os, torch
from enum import Enum
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TypeOfData(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


def create_data_dirs():
    for split in ["train", "test"]:
        for cls in ["circles", "triangles"]:
            os.makedirs(f"data/{split}/{cls}", exist_ok=True)


def create_circle(type_of_data: TypeOfData, amount: int):
    for i in range(amount):
        img = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(img)
        # Stay in side
        radius = random.randint(10, 40)
        # For now we ensure the shape is inside the canva
        center = (
            random.randint(radius, 128 - radius),
            random.randint(radius, 128 - radius),
        )
        draw.circle(xy=center, radius=radius, fill="black")
        img.save(f"data/{type_of_data.value}/circles/{i}.png")


def create_triangle(type_of_data: TypeOfData, amount: int):
    for i in range(amount):
        img = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(img)
        points = [
            (random.randint(10, 118), random.randint(10, 118)),
            (random.randint(10, 118), random.randint(10, 118)),
            (random.randint(10, 118), random.randint(10, 118)),
        ]
        draw.polygon(points, fill="black")
        img.save(f"data/{type_of_data.value}/triangles/{i}.png")


class ShapeCNN(nn.Module):
    # Define the layers of the neural net
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Diminish our image size by 2, note that does make it so it need to be divisible by 2 our input
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        #  Convert our bias and weight that we got from the neurons into a 256 a number utilized mostly due to convenicnce
        # So we dont have overfitting, this applies an afiine linear transformation which is in summary
        # A fancy way transformation of our previous vector into another vector, which follows a series of guidelines
        self.fc1 = nn.Linear(in_features=(64 * 16 * 16), out_features=256)
        self.fc2 = nn.Linear(256, 2)

    # How tensors are treated as we move along the net neurons
    def forward(self, x):
        # Here we are in summary ignoring negative values and making positive values variant
        # A good metaphor is thinking that we are only getting the good tastes and not the bad ones
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # This immediately creates the layer and inputs the x given by our neurons layers
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_loop(loader, model, loss_fn, optimizer, batch_size=64):
    size = len(loader.dataset)
    # You need to tell the model he is in training mode
    model.train()
    for batch, (images, labels) in enumerate(loader):
        # Mode the data into the device
        images = images.to(device)
        labels = labels.to(device)

        # Attempt at predict - Learning part
        pred = model(images)
        loss = loss_fn(pred, labels)
        # Backpropagate - THE CHAIN RULE
        loss.backward()
        # Applies the gradients found by backpropagation
        optimizer.step()
        # Reset to not resum
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss = loss.item()
            at = batch * batch_size
            print(f"Your current loss {loss:>7f} at instance {at} size {size}")


def eval_loop(loader, model, loss_fn):
    model.eval()
    size = len(loader.dataset)
    num_batches = len(loader)
    # Ensures we dont change the gradients while testing
    with torch.no_grad():
        test_loss, correct = 0, 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    # Quite an essential step defines the usage of our gpu
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    # Create test and train data
    if not os.path.isdir("data") or not os.listdir("data"):
        create_data_dirs()
        create_circle(TypeOfData.TRAIN, 5000)
        create_triangle(TypeOfData.TRAIN, 5000)
        create_circle(TypeOfData.TEST, 1000)
        create_triangle(TypeOfData.TEST, 1000)

    # Convert into Pytorch data objects
    train_dataset = datasets.ImageFolder("data/train", transform=transforms.ToTensor())
    test_dataset = datasets.ImageFolder("data/test", transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create the model and move it to the gpu
    model = ShapeCNN().to(device)
    # Create it is loss function and define it is optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and validate
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(
            loader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer
        )
        eval_loop(loader=test_loader, model=model, loss_fn=loss_fn)
