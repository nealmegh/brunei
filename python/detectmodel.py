import json
from threading import active_count
import matplotlib.pyplot as plt
import numpy as np
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
import pprint as pp
import matplotlib.patches as patches
from intersects import intersects
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from exif import Image as EXIMAGE
import numpy as np
from copy import deepcopy
import pprint as pp

# polfile = open(f'Sorted/Other/DJI_0213.JPG.json','r')
# jpolys = json.load(polfile)


# At 100m elevation covers approx 129 metres x 96 metres.
# At 50m approx 65 x 48 metres.

# FOV = 77°
# # （X）HFOV = 64.94°
# （Y）VFOV = 51.03°

# 0123
# 0188
# 0204
# 0205
# 0213
# 0220


# print(os.listdir('Bounding Box'))
# input()

semantic_trim = 4 # 0-4
imagefile = f'Polygon/DJI_0098/DJI_0098.JPG'
modelFile = f'Models/AcaciaNorthBearing_0.86_1674070689.h5'
# modelFile = f'Models/AcaciaMaxAvg0.84.h5'
imgNo = imagefile.replace("_", " ").replace(".", " ").split(" ")[-2]
# input(imgNo)


def decimal_coords(coords, ref):
    decimal_degrees = coords[0] + coords[1] / 60 + coords[2] / 3600
    if ref == "S" or ref == "W":
        decimal_degrees = -decimal_degrees
    return decimal_degrees


def image_coordinates(image_path):
    with open(image_path, 'rb') as src:
        img = EXIMAGE(src)
    if img.has_exif:
        try:
            # img.gps_longitude
            # for i in img:
            #     pp.pprint(i)
            # input()

            coords = (decimal_coords(img.gps_latitude,
                      img.gps_latitude_ref),
                      decimal_coords(img.gps_longitude,
                      img.gps_longitude_ref))
        except AttributeError:
            print('No Coordinates')
    else:
        print('The Image has no EXIF information')
    print(
        f"Image {src.name}, OS Version:{img.get('software', 'Not Known')} ------")
    print(f"Was taken: {img.datetime_original}, and has coordinates:{coords}")


image_coordinates(imagefile)
# input()
# info = img.getexif()

# exif_table = {}
# for tag, value in info.items():
#     decoded = TAGS.get(tag,tag)
#     exif_table[decoded] = value

# gps_info = {}

# print(exif_table['GPSInfo'])
# input()
# for key in exif_table['GPSInfo'].keys():
#     decode = GPSTAGS.get(key,key)
#     gps_info[decode] = exif_table['GPSInfo'][key]

# print(gps_info)
img = Image.open(imagefile)
img.load()

data = np.asarray(img, dtype="int32")
datacpy = deepcopy(data)
# datacpy = np.asarray(datacpy, dtype="float32")
# datacpy = datacpy / 255.0
data = data / 255.0

maxy = len(data)
maxx = len(data[0])
imgArr = []

for y in range(0, maxy, 96):
    for x in range(0, maxx, 96):
        try:
            d = deepcopy(data[y:y+96, x:x+96])
            imgArr.append(d)
        except:
            pass

imgArr = np.array(imgArr)
imgArr = np.asarray(imgArr).astype(np.float32)

print(data.shape)
print(imgArr.shape)
# input()


# a = np.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
# print('true is')
# print(a)
# print('shape is')
# print(a.shape)
# a = a.reshape(4, 3)
# print('flattened is')
# print(a)
# print('shape is')
# print(a.shape)
# input()


# modelFile = f'Models/AcaciaProper0.811671212892.h5'


# def createModel():
#     model = tfp.keras.Sequential([
#         tfp.keras.layers.InputLayer(input_shape=(96, 96, 3)),
#         tfp.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#         tfp.keras.layers.MaxPooling2D(),
#         tfp.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#         tfp.keras.layers.MaxPooling2D(),
#         tfp.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#         tfp.keras.layers.AveragePooling2D(),
#         tfp.keras.layers.Flatten(),
#         tfp.keras.layers.Dense(128, activation='relu'),
#         tfp.keras.layers.Dense(2)
#     ])
#     return model


model = tf.keras.models.load_model(modelFile)

model.summary()

probability_model = keras.Sequential([model,
                                      tf.keras.layers.Softmax()])

predictions = probability_model.predict(imgArr)



print(predictions[:10])
# for i in predictions:
#     print(np.argmax(i))
# print(test_labels[0])

datacpy = np.asarray(datacpy, dtype='float32')
datacpy1 = np.asarray(datacpy, dtype='float32')
datacpy2 = np.asarray(datacpy, dtype='float32')

p2 = predictions.reshape((int(maxy/96), int(maxx/96), 2))

linewidth = 14
coverage = []
covergae_filtered = []

for y in range(0, maxy, 96):
    for x in range(0, maxx, 96):
        try:
            if np.argmax(p2[int(y/96)][int(x/96)]) == 0:
                occupied_bounds = [False, False, False, False]

                try:
                    if np.argmax(p2[int(y/96) - 1][int(x/96)]) == 1:
                        occupied_bounds[0] = True
                except:
                    pass

                try:
                    if np.argmax(p2[int(y/96) + 1][int(x/96)]) == 1:
                        occupied_bounds[1] = True
                except:
                    pass

                try:
                    if np.argmax(p2[int(y/96)][int(x/96) - 1]) == 1:
                        occupied_bounds[2] = True
                except:
                    pass

                try:
                    if np.argmax(p2[int(y/96)][int(x/96) + 1]) == 1:
                        occupied_bounds[3] = True
                except:
                    pass

                coverage.append([int(x+96/2), int(y+96/2)])
                # print(occupied_bounds.count(False))
                if occupied_bounds.count(False) < semantic_trim:
                    covergae_filtered.append([int(x+96/2), int(y+96/2)])
                else:

                    # diag tl-br
                    for pi in range(0, 90):
                        # for xi in range(0, 95):
                        datacpy1[y+pi, x+pi] = [255.0, 0.0, 0.0]
                        datacpy1[y+pi, x+pi+1] = [255.0, 0.0, 0.0]
                        datacpy1[y+pi, x+pi+2] = [255.0, 0.0, 0.0]
                        datacpy1[y+pi, x+pi+3] = [255.0, 0.0, 0.0]
                        datacpy1[y+pi, x+pi+4] = [255.0, 0.0, 0.0]
                        datacpy1[y+pi, x+pi+5] = [255.0, 0.0, 0.0]

                    # diag bl-tr
                    for pi in range(0, 90):
                        # for xi in range(0, 95):
                        datacpy1[96+y-pi, x+pi] = [255.0, 0.0, 0.0]
                        datacpy1[96+y-pi+1, x+pi] = [255.0, 0.0, 0.0]
                        datacpy1[96+y-pi+2, x+pi] = [255.0, 0.0, 0.0]
                        datacpy1[96+y-pi+3, x+pi] = [255.0, 0.0, 0.0]
                        datacpy1[96+y-pi+4, x+pi] = [255.0, 0.0, 0.0]
                        datacpy1[96+y-pi+5, x+pi] = [255.0, 0.0, 0.0]

                    # top
                    try:
                        if occupied_bounds[0]:
                            for yi in range(y, y+linewidth):
                                for xi in range(x, x+95):
                                    datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                    except:
                        pass

                    # bottom
                    try:
                        if occupied_bounds[1]:
                            for yi in range(y+96-linewidth, y+96):
                                for xi in range(x, x+95):
                                    datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                    except:
                        pass

                    # left
                    try:
                        if occupied_bounds[2]:
                            for xi in range(x, x+linewidth):
                                for yi in range(y, y+96):
                                    datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                    except:
                        pass

                    # right
                    try:
                        if occupied_bounds[3]:
                            for xi in range(x+96-linewidth, x+96):
                                for yi in range(y, y+96):
                                    datacpy1[yi, xi] = [255.0, 0.0, 0.0]
                    except:
                        pass

        except:
            pass


###### SUBSTITUTE THE ABOVE LINES instead of for-loop ######

tcoverage = maxx * maxy / 96 / 96
# print(maxx, maxy, tcoverage)
print("coverage", len(coverage), round(len(coverage) / tcoverage, 2))
print("coverage_filtered", len(coverage) - len(covergae_filtered),
      round((len(coverage) - len(covergae_filtered)) / tcoverage, 2))


datacpy1 = datacpy1 / 255.0
plt.imsave(f'Results/Images/{imgNo}_coverage_{round(len(coverage) / tcoverage, 2)}.png', datacpy1)

imgplot = plt.imshow(datacpy1)
plt.tight_layout()
plt.show()

pp.pprint(covergae_filtered)


# datacpy2 = datacpy2 / 255.0
# plt.imsave('Results/coverage_filtered.png', datacpy1)

# imgplot = plt.imshow(datacpy2)
# plt.tight_layout()
# plt.show()
