import keras
from keras.layers import *


def ctc_loss(args):
    y_true_, label_length_, y_pre = args
    y_pre = y_pre[:, 2:, :]
    shape = K.shape(y_pre)
    inputs_length_ = K.ones([shape[0], 1]) * K.cast(shape[1], "float")
    return K.ctc_batch_cost(y_true_, y_pre, inputs_length_, label_length_)


def ctc_decode(out):
    out = out[:, 2:, :]
    shape = K.shape(out)
    return K.ctc_decode(out, input_length=K.ones(shape[0]) * K.cast(shape[1], "float"))[0][0]


class ConvWithRelu(Conv2D):
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(filters, kernel_size, kernel_initializer='he_normal', **kwargs)
        self.batch = BatchNormalization(name=self.name + "_bn")
        self.relu = Activation("relu", name=self.name + "_ac")

    def call(self, inputs):
        out = super().call(inputs)
        out = self.batch(out)
        out = self.relu(out)
        return out


def get_model(width, height, channel_num, max_string_length, char_num, is_train=True):
    inputs = keras.layers.Input([height, width, channel_num], name="img")
    output = ConvWithRelu(64, kernel_size=(3, 3), padding="same", name="block1_conv")(inputs)
    output = MaxPooling2D((2, 2), name="block1_mp")(output)
    output = ConvWithRelu(128, kernel_size=(3, 3), padding="same", name="block2_conv")(output)
    output = MaxPooling2D((2, 2), name="block2_mp")(output)
    output = ConvWithRelu(256, kernel_size=(3, 3), padding="same", name="block3_conv1")(output)
    output = ConvWithRelu(256, kernel_size=(3, 3), padding="same", name="block3_conv2")(output)
    output = MaxPooling2D((2, 1), name="block3_mp")(output)
    output = ConvWithRelu(512, kernel_size=(3, 3), padding="same", name="block3_conv")(output)
    output = BatchNormalization(name="block3_bn")(output)
    output = ConvWithRelu(512, kernel_size=(3, 3), padding="same", name="block4_conv")(output)
    output = MaxPooling2D((2, 1), name="block4_mp")(output)
    output = ConvWithRelu(512, kernel_size=(2, 2), name="block5_conv")(output)
    output = Reshape([-1, 512], name="block5_rp")(output)
    output = Bidirectional(GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1), name="block6_rnn1")(
        output)
    output = BatchNormalization(name="block6_bn")(output)
    output = Bidirectional(GRU(256, return_sequences=True, dropout=0.1, recurrent_dropout=0.1), name="block6_rnn2")(
        output)
    output = BatchNormalization(name="block6_bn2")(output)
    output = Dropout(0.2, name="block6_drop")(output)
    output = Dense(128, activation="relu", name="out_dense")(output)
    output = Dropout(0.1, name="out_drop")(output)
    output = Dense(char_num + 1, activation="softmax", name="out")(output)
    if is_train:
        label_length = keras.layers.Input([1], name="label_input")
        y_true = keras.Input([max_string_length], name="y_input")
        output = Lambda(ctc_loss, output_shape=(1,), name="zero")([y_true, label_length, output])
        model = keras.Model(inputs=[inputs, label_length, y_true], outputs=[output])
        return model
    else:
        output = Lambda(ctc_decode)(output)
        model = keras.Model(inputs=[inputs], outputs=[output])
        return model


if __name__ == "__main__":
    get_model(128, 32, 1, 10, 37).summary()
