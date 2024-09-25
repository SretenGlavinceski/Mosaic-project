import numpy as np
import cv2
import os



def get_histogram(img):

    hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])


    hist_b /= hist_b.sum() if hist_b.sum() != 0 else 1
    hist_g /= hist_g.sum() if hist_g.sum() != 0 else 1
    hist_r /= hist_r.sum() if hist_r.sum() != 0 else 1


    return hist_b.astype(np.float32).flatten(), hist_g.astype(np.float32).flatten(), hist_r.astype(np.float32).flatten()



def compare_histograms(hist1, hist2):
    hist_b1, hist_g1, hist_r1 = hist1
    hist_b2, hist_g2, hist_r2 = hist2


    hist_b1 = np.array(hist_b1, dtype=np.float32)
    hist_g1 = np.array(hist_g1, dtype=np.float32)
    hist_r1 = np.array(hist_r1, dtype=np.float32)
    hist_b2 = np.array(hist_b2, dtype=np.float32)
    hist_g2 = np.array(hist_g2, dtype=np.float32)
    hist_r2 = np.array(hist_r2, dtype=np.float32)


    correlation_b = cv2.compareHist(hist_b1, hist_b2, cv2.HISTCMP_CORREL)
    correlation_g = cv2.compareHist(hist_g1, hist_g2, cv2.HISTCMP_CORREL)
    correlation_r = cv2.compareHist(hist_r1, hist_r2, cv2.HISTCMP_CORREL)


    return (correlation_b + correlation_g + correlation_r) / 3


def get_closest_image(hist, hist_data):
    best_score = -1
    closest_image = None

    for image_path, color_hist in hist_data.items():
        score = compare_histograms(hist, color_hist)
        if score > best_score:
            best_score = score
            closest_image = image_path

    return closest_image



imgs_dir = 'images'


images = [os.path.join(root, file) for root, dirs, files in os.walk(imgs_dir) for file in files if file.endswith('.jpg')]

hist_data = {}
for img_path in images:
    img = cv2.imread(img_path)
    if img is not None:
        hist = get_histogram(img)
        hist_data[img_path] = hist


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
            tile_img = img[y0:y1, x0:x1]
            if tile_img.shape[0] != tile_height or tile_img.shape[1] != tile_width:
                print(f"Skipping tile at ({y0}, {x0}) due to shape mismatch")
                continue

            tile_hist = get_histogram(tile_img)
        except Exception as e:
            print(f"Error processing tile at ({y0}, {x0}): {e}")
            continue

        closest_image_path = get_closest_image(tile_hist, hist_data)

        if closest_image_path is None:
            continue

        i = cv2.imread(closest_image_path)
        i = cv2.resize(i, (tile_width, tile_height))
        img[y0:y1, x0:x1] = i

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    cv2.imwrite("outputHistogram.jpg", img)
else:
    print("Failed to load the image")