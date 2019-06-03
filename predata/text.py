import pickle

import tqdm
from scipy.io import loadmat

char_set = set()
train_path = "../resources/data/trainCharBound.mat"
test_path = "../resources/data/testCharBound.mat"
max_string_length = []


def get_char(data_):
    tr = tqdm.tqdm(range(data_.shape[0]))
    tr.set_description("读取图片")
    global max_string_length
    for i in tr:
        word = data_[i][1][0]
        max_string_length.append(len(word))
        for char_ in word:
            char_set.add(char_.lower())


data = loadmat(train_path)['trainCharBound'][0]
get_char(data)
data = loadmat(test_path)['testCharBound'][0]
get_char(data)

char2id = {"-": 0}

with open("../resources/chars.txt", mode="w", encoding="utf-8") as fp:
    for ind, char in enumerate(char_set):
        fp.write(char)
        fp.write("\n")
        char2id[char] = ind + 1

with open("../resources/char2id.plk", mode="wb") as fp:
    pickle.dump(char2id, fp)

print(char2id)
print(sum(max_string_length) / len(max_string_length))
print(max(max_string_length))
print(min(max_string_length))
