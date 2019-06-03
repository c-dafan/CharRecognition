import pickle
import matplotlib.pyplot as plt
import tqdm
from keras.layers import *
from keras.preprocessing.image import random_brightness
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence
from scipy.io import loadmat

from util import get_img


class ImageGenerator(Sequence):
    def __init__(self, char2id_path, train_mat, path, train_size=0.9,
                 batch_size_=16, width=128, height=32, max_string_length=15):
        self.data = loadmat(train_mat)['trainCharBound'][0]
        self.batch_size = batch_size_
        self.path = path
        self.width = width
        self.height = height
        self.max_string_length = max_string_length
        self.del_len()
        np.random.shuffle(self.data)
        self.train_size = int(self.data.shape[0] * train_size)
        with open(char2id_path, mode="rb") as fp:
            self.char2id = pickle.load(fp)

    def del_len(self):
        wls = []
        tr = tqdm.tqdm(range(self.data.shape[0]))
        tr.set_description("过滤")
        for i in tr:
            word = self.data[i][1][0]
            wls.append(len(word))
        mask = np.array(wls) <= self.max_string_length
        self.data = self.data[mask]

    def __word_process(self, word):
        return [self.char2id[w.lower()] for w in word]

    def __getitem__(self, index):
        data = self.data[index * self.batch_size: (index + 1) * self.batch_size]
        words = []
        images = []
        nums = []
        for i in range(self.batch_size):
            img_path = data[i][0][0]
            word = data[i][1][0]
            word = self.__word_process(word)
            img = get_img("{}/{}".format(self.path, img_path), self.width, self.height)
            img = random_brightness(img[:, :, np.newaxis], [0.1, 1.5])
            images.append(img[:, :, 0])
            words.append(word)
            nums.append(len(word))
        words = pad_sequences(words, maxlen=self.max_string_length, truncating="post", padding="post")
        label_input = np.array(nums).reshape([-1, 1])
        images = np.array(images)
        return {"img": images[:, :, :, np.newaxis], "label_input": label_input,
                "y_input": words}, np.zeros([self.batch_size, 1])

    def __len__(self):
        return self.train_size // self.batch_size

    def test_len(self):
        return (self.data.shape[0] - self.train_size) // self.batch_size

    def test_generator(self):
        test_len = self.test_len()
        index = 0
        while True:
            data = self.data[
                   index * self.batch_size + self.train_size: (index + 1) * self.batch_size + self.train_size]
            index += 1
            index %= test_len
            words = []
            images = []
            nums = []
            for i in range(self.batch_size):
                img_path = data[i][0][0]
                word = data[i][1][0]
                nums.append(len(word))
                word = self.__word_process(word)
                img = get_img("{}/{}".format(self.path, img_path), self.width, self.height)
                images.append(img)
                words.append(word)

            words = pad_sequences(words, maxlen=self.max_string_length, truncating="post", padding="post")
            label_input = np.array(nums).reshape([-1, 1])
            images = np.array(images)
            # images = np.transpose(images, [0, 1, 2])
            yield {"img": images[:, :, :, np.newaxis], "label_input": label_input,
                   "y_input": words}, np.zeros([self.batch_size, 1])


if __name__ == "__main__":
    path = "../resources/data/"
    train_mat = "../resources/data/trainCharBound.mat"
    char2id_path = "../resources/char2id.plk"
    img_gen = ImageGenerator(char2id_path, train_mat, path, max_string_length=10)
    id2word = dict(zip(img_gen.char2id.values(), img_gen.char2id.keys()))
    print(img_gen.data.shape)
    print(img_gen.train_size)
    print(img_gen.test_len())
    # da = img_gen.__getitem__(0)
    da = next(img_gen.test_generator())
    img = da[0]['img']
    words = da[0]['y_input']
    label_input = da[0]['label_input']

    for i in range(img.shape[0]):
        print([id2word[c] for c in words[i]])
        print(label_input[i])
        plt.imshow(img[i, :, :, 0], cmap="gray")
        plt.show()
