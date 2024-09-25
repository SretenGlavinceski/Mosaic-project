import math
import pathlib
import json
import os
import random
import numpy as np
import cv2
import time

def get_average_color(img):
    if img.size == 0:
        return None
    average_color = np.average(np.average(img, axis=0), axis=0)
    average_color = np.around(average_color, decimals=-1)
    average_color = tuple(int(i) for i in average_color)
    return average_color

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

start_time = time.time()

if "cache.json" not in os.listdir():
    imgs_dir = pathlib.Path('images')
    images = list(imgs_dir.glob("**/*.jpg"))
    data = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            average_color = get_average_color(img)
            if average_color is None:
                continue
            if str(tuple(average_color)) in data:
                data[str(tuple(average_color))].append(str(img_path))
            else:
                data[str(average_color)] = [str(img_path)]
    with open("cache.json", "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    print("Caching done")
else:
    print("Cache file already exists")

with open("cache.json", "r") as file:
    data = json.load(file)

img = cv2.imread('Mona_Lisa.jpg')
if img is not None:
    img_height, img_width, _ = img.shape
    tile_height, tile_width = 20, 20
    num_tiles_h, num_tiles_w = img_height // tile_height, img_width // tile_width
    img = img[:tile_height * num_tiles_h, :tile_width * num_tiles_w]
    tiles = []
    for y in range(0, img_height, tile_height):
        for x in range(0, img_width, tile_width):
            tiles.append((y, y + tile_height, x, x + tile_width))

    for tile in tiles:
        y0, y1, x0, x1 = tile
        tile_img = img[y0:y1, x0:x1]
        average_color = get_average_color(tile_img)
        if average_color is None:
            continue
        closest_color = get_closest_color(average_color, data.keys())
        if closest_color is None:
            continue
        i_path = random.choice(data[str(closest_color)])
        i = cv2.imread(i_path)
        if i is not None:
            i = cv2.resize(i, (tile_width, tile_height))
            img[y0:y1, x0:x1] = i
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cv2.imwrite("main.jpg", img)
else:
    print("Failed to load the image")


end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
