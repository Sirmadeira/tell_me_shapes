import random, os, torch
from enum import Enum
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ignite.engine import (
    Engine,
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine


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
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(64)

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # This immediately creates the layer and inputs the x given by our neurons layers
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# if __name__ == "__main__":
#     # Quite an essential step defines the usage of our gpu
#     device = (
#         torch.accelerator.current_accelerator().type
#         if torch.accelerator.is_available()
#         else "cpu"
#     )
#     print(f"Using {device} device")
#     # Create test and train data
#     if not os.path.isdir("data") or not os.listdir("data"):
#         create_data_dirs()
#         create_circle(TypeOfData.TRAIN, 5000)
#         create_triangle(TypeOfData.TRAIN, 5000)
#         create_circle(TypeOfData.TEST, 1000)
#         create_triangle(TypeOfData.TEST, 1000)

#     # Convert into Pytorch data objects
#     transform = transforms.Compose(
#         [
#             # Currently this mode does not care about color - We are gonna keep generating rgb because who knows
#             transforms.Grayscale(),
#             # Convert [0,255] to [0,1]
#             transforms.ToTensor(),
#             # Convert to [-1,1] which in turn makes shape detection stronger. Mote - In thesis I should calculate this before hand but I mean batch norm should apply after a while
#             transforms.Normalize(mean=0.5, std=0.5),
#         ]
#     )
#     train_dataset = datasets.ImageFolder("data/train", transform=transform)
#     test_dataset = datasets.ImageFolder("data/test", transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     # Create the model and move it to the gpu
#     model = ShapeCNN().to(device)
#     # Create it is loss function and define it is optimizer
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     # Train and validate - Note ignite this
#     # epochs = 5
#     # for t in range(epochs):
#     #     print(f"Epoch {t+1}\n-------------------------------")
#     #     train_loop(
#     #         loader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer
#     #     )
#     #     eval_loop(loader=test_loader, model=model, loss_fn=loss_fn)

#     # This is an Engine, an engine is in summary an iterator that sends events
#     trainer = create_supervised_trainer(
#         model=model, optimizer=optimizer, loss_fn=loss_fn, device=device
#     )
#     val_metrics = {"accuracy": Accuracy(), "loss": Loss(loss_fn)}

#     # Note you can also hard code this, just wrap your train_loop and eval_loop with Engine
#     train_evaluator = create_supervised_evaluator(
#         model, metrics=val_metrics, device=device
#     )
#     val_evaluator = create_supervised_evaluator(
#         model, metrics=val_metrics, device=device
#     )

#     log_interval = 100

#     @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
#     def log_training_loss(engine):
#         print(
#             f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}"
#         )

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_training_results(trainer):
#         train_evaluator.run(train_loader)
#         metrics = train_evaluator.state.metrics
#         print(
#             f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
#         )

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def log_validation_results(trainer):
#         val_evaluator.run(test_loader)
#         metrics = val_evaluator.state.metrics
#         print(
#             f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}"
#         )

#     # Checkpoint to store n_saved best models wrt score function
#     model_checkpoint = ModelCheckpoint(
#         "checkpoint",
#         n_saved=2,
#         filename_prefix="best",
#         score_function=score_function,
#         score_name="accuracy",
#         global_step_transform=global_step_from_engine(
#             trainer
#         ),  # helps fetch the trainer's state
#     )

#     # Save the model after every epoch of val_evaluator is completed
#     val_evaluator.add_event_handler(
#         Events.COMPLETED, model_checkpoint, {"model": model}
#     )

#     # Define a Tensorboard logger
#     tb_logger = TensorboardLogger(log_dir="tb-logger")

#     # Attach handler to plot trainer's loss every 100 iterations
#     tb_logger.attach_output_handler(
#         trainer,
#         event_name=Events.ITERATION_COMPLETED(every=log_interval),
#         tag="training",
#         output_transform=lambda loss: {"batch_loss": loss},
#     )

#     # Attach handler for plotting both evaluators' metrics after every epoch completes
#     for tag, evaluator in [
#         ("training", train_evaluator),
#         ("validation", val_evaluator),
#     ]:
#         tb_logger.attach_output_handler(
#             evaluator,
#             event_name=Events.EPOCH_COMPLETED,
#             tag=tag,
#             metric_names="all",
#             global_step_transform=global_step_from_engine(trainer),
#         )

#     trainer.run(train_loader, max_epochs=5)
