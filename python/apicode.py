import json
from threading import active_count
# import matplotlib.pyplot as plt
# import numpy as np
import PIL
import os
import random
import tensorflow as tf
import tensorflow.python as tfp
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Flatten, Input, Softmax
from tensorflow.python.keras.callbacks import TensorBoard
import keras

from dis import show_code
import json
from operator import truediv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import pprint as pp
import matplotlib.patches as patches
# from intersects import intersects
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from exif import Image as EXIMAGE
import numpy as np
from copy import deepcopy
import pprint as pp
import math


# print('50 x 84', 50 * math.tan(84/2 * math.pi / 180))
# print('80 x 65.5', 80 * math.tan(65.5/2 * math.pi / 180))
# print('80 x 45.5', 80 * math.tan(55/2 * math.pi / 180))
# input()


# camera_settings = {
#     'hasselblad': 84,
#     'l1d-20c': 65.5
# }

# tag_name = 'A1'
model_file = f"python/Models/AcaciaNorthBearing_0.86_1674070689.h5"
image_folder = f"Images"

# south_bearing = ['DJI_0098', 'DJI_0110',
#                  'DJI_0241', 'DJI_244', 'DJI_142', 'DJI_249', 'DJI_262', 'DJI_381', 'DJI_405', 'DJI_401', 'DJI_484']


south_bearing = {
    "A1": [(96, 99), (109, 122), (137, 150), (160, 163)],
    "A2": [(232, 232), (240, 252), (269, 280), (288, 288)],
    "A3": [(358, 359), (368, 381), (398, 408), (415, 415)],
    "A4": [(475, 476), (484, 495), (510, 520), (529, 529)],
    "A5": [(578, 580), (589, 600), (614, 624), (632, 634)],
    "A6": [(687, 688), (696, 707), (723, 737), (750, 756)],
    "A7": [(809, 812), (823, 835), (850, 863), (873, 876)],
}


# print( sum([[x for x in range(y[0], y[1]+1)] for y in south_bearing['A2']], []))
# input()


camera_settings = {
    "A1": {"fov": 84, "ralt": 50},
    "A2": {"fov": 84, "ralt": 50},
    "A3": {"fov": 84, "ralt": 50},
    "A4": {"fov": 84, "ralt": 50},
    "A5": {"fov": 84, "ralt": 50},
    "A6": {"fov": 84, "ralt": 50},
    "A7": {"fov": 65.5, "ralt": 80},
}

# polfile = open(f'Sorted/Other/DJI_0213.JPG.json','r')
# jpolys = json.load(polfile)

# 5472 x 3648 px (3:2)

# FOv = 84°
# （X）HFOV = 69.9°
# （Y）VFOV = 46.6°


# fov = 84
# cam_height = 50


img_width = 5472
img_height = 3648

width_ratio = 3
length_ratio = 2
hypotenuse = (width_ratio**2 + length_ratio**2) ** 0.5

# semantic_segment_trim = 4  # 0-4


#########################################


geo_boundaries = {"type": "FeatureCollection", "features": []}

feature = {
    "type": "Feature",
    "geometry": {"type": "Polygon", "coordinates": [[]]},
    "properties": {"title": ""},
}


def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def image_coordinates(image_path):
    with open(image_path, "rb") as src:
        img = EXIMAGE(src)
    if img.has_exif:
        try:
            # img.gps_longitude
            # for i in img:
            #     pp.pprint(img[i])
            # pp.pprint(dir(img))
            # input("done")

            coords = (
                decimal_coords(img.gps_latitude, img.gps_latitude_ref),
                decimal_coords(img.gps_longitude, img.gps_longitude_ref),
            )
        except AttributeError:
            print("No Coordinates")
    else:
        print("The Image has no EXIF information")
    print(f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
    # print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")

    return (coords[0], coords[1], img.datetime_original)


#########################################


# folders = os.listdir(image_folder)
# input(folders)

# south_bearing = images

# print(images)
# input()


# def detect_images(imagefile=f"Images/A1/Autonomous Flight/DJI_0096.JPG") -> object:
def detect_images(imagefile=Image) -> object:
    model = tf.keras.models.load_model(model_file, compile=False)
    print('hi')
    model.summary()
    probability_model = keras.Sequential([model, tf.keras.layers.Softmax()])
    print(os.path.join(os.path.abspath(os.path.dirname(__file__))))
    rotate_180 = False

    ratio_coeff = camera_settings["A1"]["fov"] / hypotenuse
    hfov = width_ratio * ratio_coeff
    lfov = length_ratio * ratio_coeff

    h_distance = camera_settings["A1"]["ralt"] * math.tan(hfov / 2 * math.pi / 180) * 2
    l_distance = camera_settings["A1"]["ralt"] * math.tan(lfov / 2 * math.pi / 180) * 2

    h_metre_per_pix = h_distance / img_width
    l_metre_per_pix = l_distance / img_height

    # print(h_metre_per_pix * 96)
    # input()

    lat_per_pix = 0.000001 * h_metre_per_pix / 0.11
    lon_per_pix = 0.000001 * l_metre_per_pix / 0.11

    # if folder not in south_bearing.keys():
    #     continue

    # images = f'{image_folder}/{folder}/Autonomous Flight'
    coverage_sem4 = []
    coverage_sem3 = []
    coverage_sem2 = []
    coverage_sem1 = []
    image_boundaries = []
    stats = {}

    # imgNo = int(imagefile.replace(
    #     "_", " ").replace(".", " ").split(" ")[-2])

    # if '.jpg' not in imagefile.lower():
    #     continue

    coords = image_coordinates(imagefile)

    # print(gps_info)
    img = Image.open(imagefile)

    if rotate_180:
        img = img.rotate(180)

    img.load()

    data = np.asarray(img, dtype="int32")
    datacpy1 = np.asarray(deepcopy(data), dtype="float32")
    datacpy1 = datacpy1 / 255.0
    data = data / 255.0

    maxy = len(data)
    maxx = len(data[0])
    imgArr = []

    for y in range(0, math.floor(maxy / 96) * 96, 96):
        for x in range(0, math.floor(maxx / 96) * 96, 96):
            try:
                d = deepcopy(data[y : y + 96, x : x + 96])
                imgArr.append(d)
            except Exception as e:
                print(e)
                # input()

    imgArr = np.array(imgArr)
    imgArr = np.asarray(imgArr).astype(np.float32)

    predictions = probability_model.predict(imgArr)
    p2 = predictions.reshape((int(maxy / 96), int(maxx / 96), 2))

    linewidth = 14
    # coverage = []
    # covergae_filtered = []

    image_boundaries = deepcopy(feature)

    image_boundaries["geometry"]["coordinates"][0] = [
        [
            coords[0] - (-0.5 * maxy) * lon_per_pix,
            coords[1] - (0.5 * maxx - maxx) * lat_per_pix,
        ],
        [
            coords[0] - (-0.5 * maxy) * lon_per_pix,
            coords[1] - (0.5 * maxx) * lat_per_pix,
        ],
        [
            coords[0] - (-0.5 * maxy + maxy) * lon_per_pix,
            coords[1] - (0.5 * maxx) * lat_per_pix,
        ],
        [
            coords[0] - (-0.5 * maxy + maxy) * lon_per_pix,
            coords[1] - (0.5 * maxx - maxx) * lat_per_pix,
        ],
        [
            coords[0] - (-0.5 * maxy) * lon_per_pix,
            coords[1] - (0.5 * maxx - maxx) * lat_per_pix,
        ],
    ]

    image_boundaries["properties"]["title"] = imagefile

    geo_boundaries["features"].append(image_boundaries)

    coverage_sem4.append([imagefile, 5, coords[0], coords[1], imagefile])

    for y in range(0, math.floor(maxy / 96) * 96, 96):
        for x in range(0, math.floor(maxx / 96) * 96, 96):
            try:
                if np.argmax(p2[int(y / 96)][int(x / 96)]) == 0:
                    occupied_bounds = [False, False, False, False]

                    try:
                        if np.argmax(p2[int(y / 96) - 1][int(x / 96)]) == 0:
                            occupied_bounds[0] = True
                    except:
                        pass

                    try:
                        if np.argmax(p2[int(y / 96) + 1][int(x / 96)]) == 0:
                            occupied_bounds[1] = True
                    except:
                        pass

                    try:
                        if np.argmax(p2[int(y / 96)][int(x / 96) - 1]) == 0:
                            occupied_bounds[2] = True
                    except:
                        pass

                    try:
                        if np.argmax(p2[int(y / 96)][int(x / 96) + 1]) == 0:
                            occupied_bounds[3] = True
                    except:
                        pass

                    # coverage.append([int(x+96/2), int(y+96/2),
                    #                 occupied_bounds.count(False)])
                    # print(occupied_bounds.count(False))

                    if occupied_bounds.count(True) < 4:
                        # covergae_filtered.append([int(x+96/2), int(y+96/2)])
                        pass
                    else:

                        # diag tl-br
                        for pi in range(0, 90):
                            # for xi in range(0, 95):
                            datacpy1[y + pi, x + pi] = [255.0, 0.0, 0.0]
                            datacpy1[y + pi, x + pi + 1] = [255.0, 0.0, 0.0]
                            datacpy1[y + pi, x + pi + 2] = [255.0, 0.0, 0.0]
                            datacpy1[y + pi, x + pi + 3] = [255.0, 0.0, 0.0]
                            datacpy1[y + pi, x + pi + 4] = [255.0, 0.0, 0.0]
                            datacpy1[y + pi, x + pi + 5] = [255.0, 0.0, 0.0]

                        # diag bl-tr
                        for pi in range(0, 90):
                            # for xi in range(0, 95):
                            datacpy1[96 + y - pi, x + pi] = [255.0, 0.0, 0.0]
                            datacpy1[96 + y - pi + 1, x + pi] = [255.0, 0.0, 0.0]
                            datacpy1[96 + y - pi + 2, x + pi] = [255.0, 0.0, 0.0]
                            datacpy1[96 + y - pi + 3, x + pi] = [255.0, 0.0, 0.0]
                            datacpy1[96 + y - pi + 4, x + pi] = [255.0, 0.0, 0.0]
                            datacpy1[96 + y - pi + 5, x + pi] = [255.0, 0.0, 0.0]

                        # top
                        try:
                            if occupied_bounds[0]:
                                for yi in range(y, y + linewidth):
                                    for xi in range(x, x + 95):
                                        datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                        except:
                            pass

                        # bottom
                        try:
                            if occupied_bounds[1]:
                                for yi in range(y + 96 - linewidth, y + 96):
                                    for xi in range(x, x + 95):
                                        datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                        except:
                            pass

                        # left
                        try:
                            if occupied_bounds[2]:
                                for xi in range(x, x + linewidth):
                                    for yi in range(y, y + 96):
                                        datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                        except:
                            pass

                        # right
                        try:
                            if occupied_bounds[3]:
                                for xi in range(x + 96 - linewidth, x + 96):
                                    for yi in range(y, y + 96):
                                        datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                        except:
                            pass

                    if occupied_bounds.count(True) == 1:
                        coverage_sem1.append(
                            [
                                imagefile,
                                1,
                                coords[0] - (-0.5 * maxy + y + 96 / 2) * lon_per_pix,
                                coords[1] - (0.5 * maxx - x + 96 / 2) * lat_per_pix,
                                "",
                            ]
                        )
                    elif occupied_bounds.count(True) == 2:
                        coverage_sem2.append(
                            [
                                imagefile,
                                2,
                                coords[0] - (-0.5 * maxy + y + 96 / 2) * lon_per_pix,
                                coords[1] - (0.5 * maxx - x + 96 / 2) * lat_per_pix,
                                "",
                            ]
                        )
                    elif occupied_bounds.count(True) == 3:
                        coverage_sem3.append(
                            [
                                imagefile,
                                3,
                                coords[0] - (-0.5 * maxy + y + 96 / 2) * lon_per_pix,
                                coords[1] - (0.5 * maxx - x + 96 / 2) * lat_per_pix,
                                "",
                            ]
                        )
                    elif occupied_bounds.count(True) == 4:
                        coverage_sem4.append(
                            [
                                imagefile,
                                4,
                                coords[0] - (-0.5 * maxy + y + 96 / 2) * lon_per_pix,
                                coords[1] - (0.5 * maxx - x + 96 / 2) * lat_per_pix,
                                "",
                            ]
                        )

                    stats["sem4"] = {
                        "coverage": len(coverage_sem4)
                        / (math.floor(maxx / 96) * math.floor(maxy / 96))
                    }
                    stats["sem3"] = {
                        "coverage": len(coverage_sem3)
                        / (math.floor(maxx / 96) * math.floor(maxy / 96))
                    }
                    stats["sem2"] = {
                        "coverage": len(coverage_sem2)
                        / (math.floor(maxx / 96) * math.floor(maxy / 96))
                    }
                    stats["sem1"] = {
                        "coverage": len(coverage_sem1)
                        / (math.floor(maxx / 96) * math.floor(maxy / 96))
                    }

            except Exception as ex:
                # return ex
                print("something wrong", ex)

    # return {stats, coverage_sem4, coverage_sem3, coverage_sem2, coverage_sem1}
    return {'a': stats, 'b': coverage_sem4, 'c': coverage_sem3, 'd': coverage_sem2, 'e': coverage_sem1}
    # datacpy1 = data
    # plt.imsave(f'Results/Images/{imgNo}_coverage_{round(len(coverage) / tcoverage, 2)}.png', datacpy1)

    # imgplot = plt.imshow(datacpy1)
    # plt.tight_layout()
    # plt.show()

    # print()
    # pp.pprint()
