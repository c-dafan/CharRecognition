from PIL import Image
import numpy as np
import random


def get_img(img_path, width, height):
    img = Image.open(img_path)
    img = img.convert("L")
    img_new = Image.new("L", (width, height), color=127)
    scale = img.height / 32
    width_ = int(img.width / scale)
    width_ *= 2
    if width_ > width:
        width_ = width
    img = img.resize((width_, height), resample=Image.BICUBIC)
    left = random.randint(0, (width - width_))
    img_new.paste(img, (left, 0))
    img_new = np.array(img_new)
    img_new = img_new / 255
    return img_new
