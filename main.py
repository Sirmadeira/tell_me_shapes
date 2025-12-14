import random
from enum import Enum
from PIL import Image, ImageDraw


class TypeOfData(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


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


if __name__ == "__main__":
    create_circle(TypeOfData.TRAIN, 5000)
    create_triangle(TypeOfData.TRAIN, 5000)
    create_circle(TypeOfData.TEST, 1000)
    create_triangle(TypeOfData.TEST, 1000)
