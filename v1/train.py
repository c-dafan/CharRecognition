import keras
import keras.backend as K

from ImageGenerator import ImageGenerator
from model import get_model

batch_size = 8
weight_path = "../resources/weight/"
path = "../resources/data/"
train_mat = "../resources/data/trainCharBound.mat"
char2id_path = "../resources/char2id.plk"
log_path = "../resources/log/"
max_string_length = 10
width = 128
height = 32
char_num = 37
epochs = 150
channel_num = 1
K.clear_session()

img_gen = ImageGenerator(char2id_path, train_mat, path, batch_size_=batch_size, max_string_length=max_string_length)
callbacks = [keras.callbacks.ModelCheckpoint(weight_path + "ep_{epoch}_loss_{loss:0.3f}_val_{val_loss:0.3f}.h5",
                                             monitor="val_loss", save_weights_only=True,
                                             save_best_only=True),
             keras.callbacks.TensorBoard(log_path, batch_size=batch_size, write_graph=True)]

my_model = get_model(width, height, channel_num, max_string_length, char_num, is_train=True)
my_model.summary()
my_model.compile(keras.optimizers.Adam(5e-4), loss=lambda y_tru, y_pred: y_pred)
my_model.fit_generator(img_gen, epochs=epochs, callbacks=callbacks,
                       validation_data=img_gen.test_generator(),
                       validation_steps=img_gen.test_len())

my_model.save_weights(weight_path + "final.h5")
