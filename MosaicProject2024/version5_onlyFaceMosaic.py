import math
import pathlib
import json
import os
import random
import numpy as np
import cv2

def get_average_color(img):
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

cache_file = "cache_face.json"
if not os.path.exists(cache_file):

    imgs_dir = pathlib.Path('images')

    images = list(imgs_dir.glob("**/*.jpg"))

    data = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            average_color = get_average_color(img)

            if str(tuple(average_color)) in data:
                data[str(tuple(average_color))].append(str(img_path))
            else:
                data[str(average_color)] = [str(img_path)]

    with open(cache_file, "w") as file:
        json.dump(data, file, indent=2, sort_keys=True)

    print("Caching done")
else:
    print("Cache file already exists")

with open(cache_file, "r") as file:
    data = json.load(file)

img = cv2.imread('Mona_Lisa.jpg')
if img is not None:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    blurred_background = cv2.GaussianBlur(img, (51, 51), 0)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img_blurred = blurred_background[y:y+h, x:x+w]

        img_height, img_width, _ = face_img.shape
        tile_height, tile_width = 20, 20

        new_height = (img_height // tile_height) * tile_height
        new_width = (img_width // tile_width) * tile_width
        face_img = face_img[:new_height, :new_width]

        tiles = []
        for y0 in range(0, new_height, tile_height):
            for x0 in range(0, new_width, tile_width):
                tiles.append((y0, y0 + tile_height, x0, x0 + tile_width))

        for tile in tiles:
            y0, y1, x0, x1 = tile
            try:
                average_color = get_average_color(face_img[y0:y1, x0:x1])
            except Exception:
                continue
            closest_color = get_closest_color(average_color, data.keys())

            if closest_color is None:
                continue

            i_path = random.choice(data[str(closest_color)])
            i = cv2.imread(i_path)
            i = cv2.resize(i, (tile_width, tile_height))
            face_img[y0:y1, x0:x1] = i

        face_img_resized = cv2.resize(face_img, (w, h))
        img[y:y+h, x:x+w] = face_img_resized

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output_face_mosaic.jpg", img)
else:
    print("Failed to load the image")
