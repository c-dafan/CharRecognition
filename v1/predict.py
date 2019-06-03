import os
import pickle

import matplotlib.pyplot as plt
from keras.layers import *

from model import get_model
from util import get_img

weight_path = "../resources/weight/ep_9_loss_0.009_val_5.666.h5"
char2id_path = "../resources/char2id.plk"
max_string_length = 10
width = 128
char_num = 37
height = 32

with open(char2id_path, mode="rb") as fp:
    char2id = pickle.load(fp)

id2char = dict(zip(char2id.values(), char2id.keys()))


def predict(base_model, file_):
    img = get_img(file_, width, height)
    y_pred = base_model.predict(img[np.newaxis, :, :, np.newaxis])
    print(file_.center(50, "*"))
    print(y_pred)
    str_out = ''.join([id2char[x] for x in y_pred[0] if x != -1])
    print(str_out)
    plt.imshow(img, cmap="gray")
    plt.show()


my_model = get_model(width, height, 1, max_string_length, char_num, is_train=False)
my_model.load_weights(weight_path)

file_list = os.listdir("../resources/data/test/")
for file in file_list:
    file = os.path.join("../resources/data/test/", file)
    predict(my_model, file)
