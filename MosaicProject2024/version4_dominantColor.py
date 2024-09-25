import math
import pathlib
import json
import os
import random
import numpy as np
import cv2
from collections import Counter



def get_dominant_color(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixels = img_rgb.reshape(-1, 3)

    color_counts = Counter(map(tuple, pixels))

    dominant_color, _ = color_counts.most_common(1)[0]

    return dominant_color


def get_closest_color(color, colors):
    cr, cg, cb = color
    min_difference = float("inf")
    closest_color = None
    for c in colors:
        r, g, b = eval(c)
        difference = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if difference < min_difference:
            min_difference = difference
            closest_color = eval(c)
    return closest_color


cache_filename = "cache_dominant.json"
if cache_filename not in os.listdir():
    imgs_dir = pathlib.Path('images')
    images = list(imgs_dir.glob("**/*.jpg"))

    data = {}
    for img_path in images:

        img = cv2.imread(str(img_path))
        if img is not None:
            dominant_color = get_dominant_color(img)
            data[str(dominant_color)] = str(img_path)
            print(f'{img_path} success!')

    with open(cache_filename, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)

    print("Caching done")
else:
    print("Cache file already exists")

with open(cache_filename, "r") as file:
    data = json.load(file)


img = cv2.imread('Mona_Lisa.jpg')
if img is not None:
    img_height, img_width, _ = img.shape
    tile_height, tile_width = 5, 5
    num_tiles_h, num_tiles_w = img_height // tile_height, img_width // tile_width
    img = img[:tile_height * num_tiles_h, :tile_width * num_tiles_w]

    tiles = []
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            tiles.append((y, y + tile_height, x, x + tile_width))

    for tile in tiles:
        y0, y1, x0, x1 = tile
        try:
            dominant_color = get_dominant_color(img[y0:y1, x0:x1])
        except Exception:
            continue

        closest_color = get_closest_color(dominant_color, data.keys())
        if closest_color is None:
            continue

        i_path = data[str(closest_color)]
        i = cv2.imread(i_path)
        i = cv2.resize(i, (tile_width, tile_height))
        img[y0:y1, x0:x1] = i

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cv2.imwrite("outputDominant.jpg", img)
else:
    print("Failed to load the image")
