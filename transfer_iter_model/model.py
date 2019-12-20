import gc
import logging

import keras.losses
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input, Lambda
from keras.models import Model, load_model

from bearing.plot_lstm_feature import plot
from bearing.transfer_iter_model.data_generator import DataGenerator
from bearing.transfer_iter_model.handle_result import generate_metrics

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


def mmd(x):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)
    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    kvar = K.constant(value=np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                      dtype='float32')

    train_tensor = tf.map_fn(lambda cur: tf.cond(
        K.all(K.equal(cur[1], kvar)), lambda: K.expand_dims(
            K.zeros_like(cur[0]), axis=0), lambda: K.expand_dims(cur[0],
                                                                 axis=0)),
                             (x[0], x[2]),
                             dtype=(tf.float32))
    test_tensor = tf.map_fn(lambda cur: tf.cond(
        K.all(K.equal(cur[1], kvar)), lambda: K.expand_dims(
            K.zeros_like(cur[0]), axis=0), lambda: K.expand_dims(cur[0],
                                                                 axis=0)),
                            (x[1], x[3]),
                            dtype=(tf.float32))

    beta = 1.0
    x1x1 = gaussian_kernel(train_tensor, train_tensor, beta)
    x1x2 = gaussian_kernel(train_tensor, test_tensor, beta)
    x2x2 = gaussian_kernel(test_tensor, test_tensor, beta)
    diff = K.mean(x1x1) - 2 * K.mean(x1x2) + K.mean(x2x2)

    return diff


def gaussian_kernel(x1, x2, beta=1.0):
    # r = x1.dimshuffle(0,"x",1)
    r = K.expand_dims(x1, 1)
    return K.exp(-beta * K.sum(K.square(r - x2), axis=-1))


def get_mmd_loss(y_true, y_pred):
    return y_pred


class TransferClassSensitiveModel():
    def __init__(self, args, data_dic, one_hot_encoder, class_weights,
                 dic_path):
        self.args = args
        self.data_dic = data_dic
        self.one_hot_encoder = one_hot_encoder
        self.class_weights = class_weights
        self.dic_path = dic_path

        self.show_eval_figure = False
        self.model = None
        self.history = None

    def _model(self):
        input_train = Input(shape=(self.args.sliding_window_length, 2),
                            name='input_train')
        input_test = Input(shape=(self.args.sliding_window_length, 2),
                           name='input_test')

        input_train_label = Input(shape=(10, ), name='input_train_label')
        input_test_label = Input(shape=(10, ), name='input_test_label')

        conv_1_shared = Conv1D(10,
                               3,
                               strides=1,
                               activation='tanh',
                               name='conv_1_shared')
        conv_1_train = conv_1_shared(input_train)
        conv_1_test = conv_1_shared(input_test)
        # print(mmd_compute)

        conv_1_train_dropout = Dropout(0.3)(conv_1_train)
        conv_1_test_dropout = Dropout(0.3)(conv_1_test)

        lstm_1_shared = LSTM(self.args.hidden_size,
                             return_sequences=True,
                             activation='tanh')

        lstm_1_train = lstm_1_shared(conv_1_train_dropout)
        lstm_1_test = lstm_1_shared(conv_1_test_dropout)

        mmd_compute = Lambda(lambda x: mmd(x), name='mmd_compute')(
            [lstm_1_train, lstm_1_test, input_train_label, input_test_label])

        lstm_1_train_dropout = Dropout(0.3)(lstm_1_train)

        flatten = Flatten()(lstm_1_train_dropout)
        output = Dense(10, activation='softmax')(flatten)

        self.model = Model(inputs=[
            input_train, input_test, input_train_label, input_test_label
        ],
                           outputs=[output, mmd_compute])

        self.model.compile(loss=['categorical_crossentropy', get_mmd_loss],
                           loss_weights=[1., 0.1],
                           optimizer='rmsprop',
                           metrics={
                               'output_a': 'accuracy',
                               'output_b': None
                           })
        return self.model

    def _save_model(self):
        logger.info("Saved model...")
        self.model.save(self.dic_path +
                        "/model.h5")  # creates a HDF5 file "my_model.h5"

    def _save_figure(self, show=False):
        logger.info("Saved evaluate figure...")
        # Plot training & validation accuracy values
        plt.plot(self.history.history["acc"])
        plt.plot(self.history.history["val_acc"])

        # plt.plot(history.history["val_output_a_acc"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig(self.dic_path + "/loss.jpg")
        if show:
            plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.savefig(self.dic_path + "/accuracy.jpg")
        if show:
            plt.show()

    def _model_evaluate(self):
        '''
        to-do
        '''
        logger.info("Evaluate on test dataset...")

        test_loss, _, _, test_accuracy = self.model.evaluate([
            self.data_dic['test_feature'],
            np.zeros((self.test_feature.shape[0], self.test_feature.shape[1],
                      self.test_feature.shape[2]))
        ], [
            self.test_label,
            np.zeros((self.test_label.shape[0], self.test_label.shape[1],
                      self.test_label.shape[2]))
        ])
        logger.info(
            "predict categorical crossentropy loss : {:.5f}  predict accuracy : {:.5f}"
            .format(test_loss, test_accuracy))

    def _load_exist_model(self):
        keras.losses.get_mmd_loss = get_mmd_loss

        input_train = Input(shape=(self.args.sliding_window_length, 2),
                            name='input_train')
        input_test = Input(shape=(self.args.sliding_window_length, 2),
                           name='input_test')

        input_train_label = Input(shape=(10, ), name='input_train_label')
        input_test_label = Input(shape=(10, ), name='input_test_label')

        conv_1_shared = Conv1D(10,
                               3,
                               strides=1,
                               activation='tanh',
                               name='conv_1_shared')
        conv_1_train = conv_1_shared(input_train)
        conv_1_test = conv_1_shared(input_test)
        # print(mmd_compute)

        conv_1_train_dropout = Dropout(0.3)(conv_1_train)
        conv_1_test_dropout = Dropout(0.3)(conv_1_test)

        lstm_1_shared = LSTM(self.args.hidden_size,
                             return_sequences=True,
                             activation='tanh')

        lstm_1_train = lstm_1_shared(conv_1_train_dropout)
        lstm_1_test = lstm_1_shared(conv_1_test_dropout)

        def mmd(item):
            return K.mean(item[0])

        mmd_compute = Lambda(lambda x: mmd(x), name='mmd_compute')(
            [lstm_1_train, lstm_1_test, input_train_label, input_test_label])

        logger.info("Loading exist model...")
        self.model = load_model(self.dic_path + '/model.h5',
                                custom_objects={
                                    'mmd_compute': mmd_compute,
                                    'mmd': mmd,
                                })

    def _get_predict_result_and_middle_feature(self):
        logger.info("Predict result on train and test dataset and saved...")

        print(self.data_dic['train_feature'].shape)

        train_predict_result = self.model.predict([
            self.data_dic['train_feature'],
            self.data_dic['test_feature_for_transfer'],
            self.data_dic['train_label'],
            self.data_dic['test_label_for_transfer']
        ])

        print("train has been predicted ...")
        print(self.data_dic['test_feature'].shape,
              self.data_dic['test_feature_for_transfer'].shape,
              self.data_dic['test_label'].shape,
              self.data_dic['test_label_for_transfer'].shape)
        test_predict_result = self.model.predict([
            self.data_dic['test_feature'], self.data_dic['test_feature'],
            self.data_dic['test_label'], self.data_dic['test_label']
        ])

        print("test has been predicted ...")
        np.save(self.dic_path + "/train_label_encoder.npy",
                self.data_dic['train_label'])

        np.save(self.dic_path + "/train_predict_result.npy",
                train_predict_result[0])
        np.save(self.dic_path + "/test_predict_result.npy",
                test_predict_result[0])

        # raw result

        train_predict_result_inverse = self.one_hot_encoder.inverse_transform(
            train_predict_result[0])
        test_predict_result_inverse = self.one_hot_encoder.inverse_transform(
            test_predict_result[0])

        train_predict_result_inverse = np.reshape(
            train_predict_result_inverse,
            (train_predict_result_inverse.shape[0]))
        test_predict_result_inverse = np.reshape(
            test_predict_result_inverse,
            (test_predict_result_inverse.shape[0]))

        np.save(self.dic_path + "/train_predict_result_inverse.npy",
                train_predict_result_inverse)
        np.save(self.dic_path + "/test_predict_result_inverse.npy",
                test_predict_result_inverse)

        del train_predict_result_inverse, test_predict_result_inverse
        gc.collect()

        train_label_inverse = self.one_hot_encoder.inverse_transform(
            self.data_dic['train_label'])
        train_label_inverse = np.reshape(train_label_inverse,
                                         (train_label_inverse.shape[0]))
        np.save(self.dic_path + "/train_label.npy", train_label_inverse)

        # logger.info("Get softmax layer output...")
        print(self.model.layers, self.model.layers[0].input,
              self.model.layers[3].input, self.model.layers[-2].output)

        logger.info("Get middle feature output from cnn/lstm/dense...")
        get_layer_output = K.function([
            self.model.layers[0].input, self.model.layers[3].input,
            self.model.layers[8].input, self.model.layers[9].input
        ], [
            self.model.layers[4].get_output_at(0),
            self.model.layers[-1].get_output_at(0)
        ])

        train_layer_output = get_layer_output([
            self.data_dic['train_feature'],
            self.data_dic['test_feature_for_transfer'],
            self.data_dic['train_label'],
            self.data_dic['test_label_for_transfer']
        ])
        test_layer_output = get_layer_output([
            self.data_dic['test_feature'],
            self.data_dic['test_feature_for_transfer'],
            self.data_dic['test_label'],
            self.data_dic['test_label_for_transfer']
        ])

        # lstm fature
        np.save(self.dic_path + "/train_lstm_feature.npy",
                train_layer_output[0])
        np.save(self.dic_path + "/test_lstm_feature.npy", test_layer_output[0])

        # softmax output
        np.save(self.dic_path + "/train_softmax_feature.npy",
                train_layer_output[1])
        np.save(self.dic_path + "/test_softmax_feature.npy",
                test_layer_output[1])

    def train_model(self):
        self._model()
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.args.early_stopping_patience,
            verbose=1)

        # data generator
        training_generator = DataGenerator(
            self.data_dic['train_feature'],
            self.data_dic['test_feature_for_transfer'],
            self.data_dic['train_label'],
            self.data_dic['test_label_for_transfer'], self.model,
            self.one_hot_encoder, self.args.batch_size)
        validation_generator = DataGenerator(
            self.data_dic['validation_feature'],
            self.data_dic['validation_feature_for_transfer'],
            self.data_dic['validation_label'],
            self.data_dic['validation_label_for_transfer'], None, None,
            self.args.batch_size)

        self.history = self.model.fit_generator(
            generator=training_generator,
            validation_data=validation_generator,
            use_multiprocessing=True,
            workers=6,
            epochs=self.args.epochs,
            verbose=0,
            class_weight=[self.class_weights, self.class_weights],
            callbacks=[early_stopping])

        self._save_model()
        self._load_exist_model()
        # self._save_figure(self.show_eval_figure)
        # self._model_evaluate()
        self._get_predict_result_and_middle_feature()
        generate_metrics(self.dic_path)
        plot(self.dic_path, self.args.train_motor, self.args.test_motor)

    def predict_with_exist_model(self):
        self._load_exist_model()
        self._get_predict_result_and_middle_feature()
        generate_metrics(self.dic_path)
        plot(self.dic_path, self.args.train_motor, self.args.test_motor)

    def get_model(self):
        return self.model
