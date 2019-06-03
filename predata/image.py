import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm

img_path = "../resources/data/train"
img_list = os.listdir(img_path)
shapes = []

tr = tqdm.tqdm(img_list)
tr.set_description("读取图片")
for img_name in tr:
    img_name = os.path.join(img_path, img_name)
    img = plt.imread(img_name)
    shape = [img.shape[0], img.shape[1]]
    shapes.append(shape)

shapes = np.array(shapes)
scales = shapes[:, :1] / 32
shapes = shapes / scales
scales = scales.astype(np.int)
print(shapes[:, 1].max())
print(shapes[:, 1].mean())
print((shapes[:, 1] > 200).sum())
