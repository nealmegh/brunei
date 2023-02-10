import json
from threading import active_count
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import datetime
import random
import tensorflow as tf
import tensorflow.python as tfp
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Flatten, Input, Softmax
from tensorflow.python.keras.callbacks import TensorBoard, Callback
# from tf..keras.utils.vis_utils import plot_model


instance = int(datetime.datetime.now().timestamp())

print("instance is: ", instance)



class LossAndErrorPrintingCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(
            "On batch {}, avg. loss {:7.2f}.".format(batch, logs["loss"])
        )

    def on_test_batch_end(self, batch, logs=None):
        print(
            "On batch {}, avg. loss {:7.2f}.".format(batch, logs["loss"])
        )

    def on_epoch_end(self, epoch, logs=None):
        cols = list(logs.keys())
        cols.sort()
        hcols = f"epoch,{','.join(cols)},\n"
        vcols = f"{epoch},"

        for c in cols:
            vcols += f"{logs[c]},"
        vcols += "\n"

        if f"epoch_training_log_{instance}.csv" not in os.listdir('logs/'):
            with open(f"logs/epoch_training_log_{instance}.csv", "w+") as touch:
                touch.write(hcols)

        with open(f"logs/epoch_training_log_{instance}.csv", "a+") as fp:
            fp.write(vcols)


print(tf.__version__, "\n")


def acacia_load_data(train=18000, test=2000, per_img=400):

    images = {"Acacia Species": [], "Other": []}
    class_names = []

    for i, species in enumerate(os.listdir('Sorted')):
        class_names.append(species)
        lst = list(os.listdir(f'Sorted/{species}'))
        random.shuffle(lst)

        for j, file in enumerate(lst):
            if len(images[species]) >= round(train / 2):
                break

            try:
                data = json.load(open(f'Sorted/{species}/{file}', 'r'))
                random.shuffle(data)
                for img in data[:per_img]:
                    images[species].append((img, i))                
            except:
                continue

            print(
                f'loaded file {file} totals {len(images["Acacia Species"])} Acacia and {len(images["Other"])} Other')

    print(f"Acacia {len(images['Acacia Species'])} Other {len(images['Other'])}")

    random.shuffle(images["Acacia Species"])
    random.shuffle(images["Other"])

    combined = images["Acacia Species"][:(
        train + test)//2] + images["Other"][:(train + test)//2]
    random.shuffle(combined)

    train_images = np.array([x[0] for x in combined[:train]])
    train_labels = np.array([x[1] for x in combined[:train]])
    test_images = np.array([x[0] for x in combined[train:train+test]])
    test_labels = np.array([x[1] for x in combined[train:train+test]])

    return ((train_images, train_labels), (test_images, test_labels), class_names)


(train_images, train_labels), (test_images, test_labels), class_names = acacia_load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0


def createModel_tfp():
    model = tfp.keras.Sequential([
        tfp.keras.layers.InputLayer(input_shape=(96, 96, 3)),
        tfp.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tfp.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        tfp.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tfp.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        tfp.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tfp.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        tfp.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        # tfp.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        # tfp.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tfp.keras.layers.AveragePooling2D(),
        tfp.keras.layers.Flatten(),
        tfp.keras.layers.Dense(128, activation='relu'),
        tfp.keras.layers.Dense(2)
    ])
    return model


# def createModel_top():
#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),
#         tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         # tfp.keras.layers.AveragePooling2D(),
#         tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         # tfp.keras.layers.AveragePooling2D(),
#         tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         # tfp.keras.layers.AveragePooling2D(),
#         tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         # tf.keras.layers.AveragePooling2D(),
#         tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
#         tf.keras.layers.AveragePooling2D(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(2)
#     ])
#     return model


def createModel():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(96, 96, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # tfp.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    return model


model = createModel_tfp()

# model = createModel()
# tf.keras.utils.plot_model(model, to_file=f'net_model_{instance}.png',
#                           show_shapes=True, show_layer_names=False)
# print("saved model arch to [net_model.png] [enter] proceeding to training")



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy', 'mae', 'mse', 'RootMeanSquaredError'])

model.fit(train_images, train_labels, epochs=50,
          callbacks=[LossAndErrorPrintingCallback()])


print('#############')
print(model.evaluate(test_images, test_labels, verbose=2))
print('#############')

test_loss, test_acc, test_mae, test_mse, test_rmse = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


mfile = F'Models/AcaciaNorthBearing_{round(test_acc, 2)}_{instance}.h5'
model.save(mfile)
print("saved model to ", mfile)
