import logging

import keras
import numpy as np
import tensorflow as tf

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 train_feature,
                 test_feature,
                 train_label,
                 test_label,
                 model,
                 one_hot_encoder,
                 batch_size=512,
                 shuffle=True):
        'Initialization'
        self.train_feature = train_feature
        self.test_feature = test_feature
        self.train_label = train_label
        self.test_label = test_label
        self.model = model
        self.one_hot_encoder = one_hot_encoder

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # return 10
        return int(np.floor(len(self.train_feature) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # print('here')
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Generate data
        # X, y = self.__data_generation(indexes)
        return [
            self.train_feature[indexes], self.test_feature[indexes],
            self.train_label[indexes], self.test_label[indexes]
        ], [self.train_label[indexes], self.test_label[indexes]]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # if self.model:
        if self.model:
            test_predict_result = self.model.predict([
                self.test_feature, self.test_feature, self.test_label,
                self.test_label
            ])[0]

            test_predict_result = np.reshape(np.argmax(test_predict_result, axis=1), (-1, 1))
            test_predict_result = self.one_hot_encoder.transform(test_predict_result).toarray()
            print('epoch end test predict : ', test_predict_result)
            self.test_label = test_predict_result
        self.indexes = np.arange(len(self.train_feature))
        if self.shuffle:
            np.random.shuffle(self.indexes)
