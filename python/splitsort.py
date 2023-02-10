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
import numpy as np
from copy import deepcopy
import time


# print(f"{3%2}")
# input()

# 5472 x 3648

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(
        np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)

# print(os.listdir('Polygon'))


def InitFolder(folder):
    if not os.path.isdir("Sorted"):
        os.mkdir("Sorted")

    if not os.path.isdir(f"Sorted/{folder}"):
        os.mkdir(f"Sorted/{folder}")


InitFolder('Acacia Species')
InitFolder('Other')

tic = time.perf_counter()

files_total = len(os.listdir('Polygon'))
for iidx, imgfile in enumerate(os.listdir('Polygon')):
    files = [x for x in os.listdir(
        f'Polygon/{imgfile}') if 'ted.JPG' not in x and 'efiles.json' not in x and 'efile.json' not in x and 'csv' not in x]
    # input(files)

    img = load_image(f'Polygon/{imgfile}/{files[0]}')

    maxy = len(img)
    maxx = len(img[0])

    print(f' processing',files, maxy, maxx )

    polfile = open(f'Polygon/{imgfile}/{files[1]}')
    jpolys = json.load(polfile)
    headers = list(jpolys.keys())
    hidx = 0

    for idx, j in enumerate(headers):
        if imgfile in j:
            hidx = idx
    regions = jpolys[list(jpolys.keys())[idx]]['regions']

    acacia = []
    nonacacia = []

    for y in range(0, maxy, 96):
        for x in range(0, maxx, 96):
            isAcacia = False

            for r in regions:

                xs = r["shape_attributes"]["all_points_x"]
                ys = r["shape_attributes"]["all_points_y"]
                intersect_count = 0

                if intersects(((x+48, y+48), (maxx, y)), ((xs[-1], ys[-1]), (xs[0], ys[0]))) == True:
                    intersect_count += 1

                for i in range(len(xs)-1):
                    if intersects(((x+48, y+48), (maxx, y)), ((xs[i], ys[i]), (xs[i+1], ys[i+1]))) == True:
                        intersect_count += 1

                if intersect_count % 2 == 1:
                    # print(F"{files[0]} {x +48} {y+48} this is Acacia")
                    # ax = plt.gca()
                    # rect = patches.Rectangle(
                    #     (x, y), 96, 96, linewidth=1, edgecolor='r', facecolor='none')
                    # ax.add_patch(rect)
                    isAcacia = True

            img_tile = deepcopy(img[y:y+96, x:x+96])
            if isAcacia:
                acacia.append(img_tile)
            else:
                nonacacia.append(img_tile)

    with open(f'Sorted/Acacia Species/{files[0]}.json', 'w') as f:
        json.dump(acacia, f, cls=NumpyEncoder)

    with open(f'Sorted/Other/{files[0]}.json', 'w') as f:
        json.dump(nonacacia, f, cls=NumpyEncoder)

    print(f"{iidx +1}/{files_total}", imgfile, f'acacia = {len(acacia)}',
          f'nonacacia = {len(nonacacia)}')

    # imgplot = plt.imshow(img)
    # plt.tight_layout()
    # plt.show()

toc = time.perf_counter()
print(f'completed in {toc-tic: 0.4f} seconds')