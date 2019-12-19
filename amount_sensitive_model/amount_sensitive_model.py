import argparse
import gc
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Conv1D, Dense, Dropout, Flatten, Input
from keras.models import Model, load_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from bearing.amount_sensitive_model.handle_result import generate_metrics
from bearing.constants import const
from bearing.plot_lstm_feature import plot

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.ERROR)

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--train-motor',
                    help='train motor index',
                    type=str,
                    default='0')

flag_parser = parser.add_mutually_exclusive_group(required=False)
flag_parser.add_argument('--flag', dest='flag', action='store_true')
flag_parser.add_argument('--no-flag', dest='flag', action='store_false')
parser.set_defaults(flag=True)

parser.add_argument('--model-dic-path',
                    help='if train-flag = True, provide a model dic path',
                    type=str,
                    default=None)
args = parser.parse_args()


def mkdir(path):

    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def get_random_block_from_test(data, size):
    sample = np.random.randint(len(data), size=size)
    data = data[sample]
    return data


class AmountSensitiveModel():
    def __init__(self, train_feature, train_label, validation_feature,
                 validation_label, test_feature, test_label, one_hot_encoder,
                 model_params, class_weights, dic_path, train_motor,
                 test_motor):
        self.train_feature = train_feature
        self.train_label = train_label
        self.validation_feature = validation_feature
        self.validation_label = validation_label
        self.test_feature = test_feature
        self.test_label = test_label
        self.one_hot_encoder = one_hot_encoder
        self.model_params = model_params
        self.class_weights = class_weights
        self.dic_path = dic_path
        self.train_motor = train_motor
        self.test_motor = test_motor

        self.show_eval_figure = False
        self.model = None
        self.history = None

    def _model(self):
        input_train = Input(shape=(const.SLIDING_WINDOW_LENGTH, 2),
                            name="input_train")

        conv_1_train = Conv1D(10,
                              3,
                              strides=1,
                              activation="tanh",
                              name="conv_1_shared")(input_train)

        conv_1_dropout = Dropout(0.3)(conv_1_train)

        lstm_1 = LSTM(self.model_params["hidden_size"],
                      return_sequences=True,
                      activation="tanh")(conv_1_dropout)
        lstm_1_dropout = Dropout(0.3)(lstm_1)

        flatten = Flatten()(lstm_1_dropout)
        output = Dense(10, activation="softmax")(flatten)

        self.model = Model(inputs=input_train, outputs=output)

        self.model.compile(loss="categorical_crossentropy",
                           optimizer="rmsprop",
                           metrics=["accuracy"])

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
        logger.info("Evaluate on test dataset...")
        test_loss, test_accuracy = self.model.evaluate([self.test_feature],
                                                       [self.test_label])

        logger.info(
            "predict categorical crossentropy loss : {:.5f}  predict accuracy : {:.5f}"
            .format(test_loss, test_accuracy))

    def _load_exist_model(self):
        logger.info("Loading exist model...")
        self.model = load_model(self.dic_path + '/model.h5')

    def _get_predict_result_and_middle_feature(self):
        logger.info("Predict result on train and test dataset and saved...")
        train_predict_result = self.model.predict(self.train_feature)
        test_predict_result = self.model.predict(self.test_feature)

        np.save(self.dic_path + "/train_predict_result.npy",
                train_predict_result)
        np.save(self.dic_path + "/test_predict_result.npy",
                test_predict_result)

        # raw result
        train_predict_result_inverse = self.one_hot_encoder.inverse_transform(
            train_predict_result)
        test_predict_result_inverse = self.one_hot_encoder.inverse_transform(
            test_predict_result)
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
            self.train_label)
        train_label_inverse = np.reshape(train_label_inverse,
                                         (train_label_inverse.shape[0]))
        np.save(self.dic_path + "/train_label.npy", train_label_inverse)

        logger.info("Get softmax layer output...")
        get_softmax_layer_output = K.function([self.model.layers[0].input],
                                              [self.model.layers[-1].output])
        train_softmax_layer_output = get_softmax_layer_output(
            [self.train_feature])
        test_softmax_layer_output = get_softmax_layer_output(
            [self.test_feature])

        # softmax output
        np.save(self.dic_path + "/train_softmax_feature.npy",
                train_softmax_layer_output)
        np.save(self.dic_path + "/test_softmax_feature.npy",
                test_softmax_layer_output)

        logger.info("Get middle feature output from cnn/lstm/dense...")
        get_layer_output = K.function([self.model.layers[0].input], [
            self.model.layers[1].output, self.model.layers[4].output,
            self.model.layers[-1].output
        ])

        train_layer_output = get_layer_output([self.train_feature])
        test_layer_output = get_layer_output([self.test_feature])
        '''
        # cnn feature
        np.save(self.dic_path + "/train_cnn_feature.npy",
                train_layer_output[0])
        np.save(self.dic_path + "/test_cnn_feature.npy", test_layer_output[0])
        '''
        # lstm fature
        np.save(self.dic_path + "/train_lstm_feature.npy",
                train_layer_output[1])
        np.save(self.dic_path + "/test_lstm_feature.npy", test_layer_output[1])
        '''
        # softmax output
        np.save(self.dic_path + "/train_softmax_feature.npy",
                train_layer_output[2])
        np.save(self.dic_path + "/test_softmax_feature.npy",
                test_layer_output[2])
        '''
    def train_model(self):
        self._model()
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.model_params["early_stopping_patience"],
            verbose=1)
        self.history = self.model.fit(
            x=self.train_feature,
            y=self.train_label,
            epochs=self.model_params["epochs"],
            verbose=self.model_params["verbose"],
            class_weight=self.class_weights,
            validation_data=(self.validation_feature, self.validation_label),
            batch_size=self.model_params["batch_size"],
            shuffle=self.model_params["shuffle"],
            callbacks=[early_stopping])
        # logger.info("History : ", self.history)
        history_logger = self.history.history
        for i in range(len(history_logger["acc"])):
            logger.info(
                "Epoch : {:<4d} loss : {:.5f} acc : {:.5f} val_loss : {:.5f} val_acc : {:.5f}"
                .format(i, history_logger["loss"][i], history_logger["acc"][i],
                        history_logger["val_loss"][i],
                        history_logger["val_acc"][i]))
        self._save_model()
        self._save_figure(self.show_eval_figure)
        self._model_evaluate()
        self._get_predict_result_and_middle_feature()
        generate_metrics(self.dic_path)
        plot(self.dic_path, self.train_motor, self.test_motor)

    def predict_with_exist_model(self):
        self._load_exist_model()
        self._get_predict_result_and_middle_feature()
        generate_metrics(self.dic_path)
        plot(self.dic_path, self.train_motor, self.test_motor)

    def get_model(self):
        return self.model


def main():
    module_path = os.path.dirname(os.path.abspath(__file__))
    pre_module_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))

    params = {
        "train_motor": args.train_motor,
        "test_motor": 3,
        "train_flag": args.flag,
        "model_dic_path": args.model_dic_path
    }
    model_params = {
        "batch_size": 512,
        "hidden_size": 32,
        "epochs": 70,
        "verbose": 0,
        "shuffle": True,
        "early_stopping_patience": 10,
    }

    if params["train_flag"]:
        # mkdir
        dic_path = "{}/saved_model/{}_cnn_lstm_sliding_{}_motor_train_{}_test_{}".format(
            module_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
            const.SLIDING_WINDOW_LENGTH, params["train_motor"],
            params["test_motor"])
        mkdir(dic_path)
        logging_file = dic_path + "/" + str(
            os.path.basename(__file__).split(".")[0]) + ".log"
    else:
        dic_path = params["model_dic_path"]
        logging_file = dic_path + "/" + str(
            os.path.basename(__file__).split(".")[0]) + "_add.log"

    logging.basicConfig(
        filename=logging_file,
        # stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w")
    # load data

    logger.info("Loading feature and label...")

    train_feature = np.load(
        "{}/new_dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy"
        .format(pre_module_path, params["train_motor"],
                const.SLIDING_WINDOW_LENGTH))
    train_label = np.load(
        "{}/new_dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy"
        .format(pre_module_path, params["train_motor"],
                const.SLIDING_WINDOW_LENGTH))

    test_feature = np.load(
        "{}/new_dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy"
        .format(pre_module_path, params["test_motor"],
                const.SLIDING_WINDOW_LENGTH))
    test_label = np.load(
        "{}/new_dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy"
        .format(pre_module_path, params["test_motor"],
                const.SLIDING_WINDOW_LENGTH))

    logger.info("train feature shape : {} train label shape : {}".format(
        train_feature.shape, train_label.shape))
    logger.info("test feature shape  : {} test label shape  : {}".format(
        test_feature.shape, test_label.shape))
    # Handle outlier
    logger.info("Handle outlier...")
    train_feature = np.nan_to_num(train_feature)
    test_feature = np.nan_to_num(test_feature)
    # One-hot encoder
    logger.info("One-hot encoder...")
    one_hot_encoder = preprocessing.OneHotEncoder(categories="auto")
    train_label = np.reshape(train_label, (train_label.shape[0], 1))
    test_label = np.reshape(test_label, (test_label.shape[0], 1))

    one_hot_encoder.fit(train_label)

    train_label_encoder = one_hot_encoder.transform(train_label).toarray()
    test_label_encoder = one_hot_encoder.transform(test_label).toarray()
    np.save(dic_path + '/train_init_label.npy', train_label)
    np.save(dic_path + '/test_label.npy', test_label)

    np.save(dic_path + '/train_init_label_encoder.npy', train_label_encoder)
    np.save(dic_path + '/test_label_encoder.npy', test_label_encoder)

    logger.info("train label encoder shape : {}".format(
        train_label_encoder.shape))
    logger.info("test label encoder shape  : {}".format(
        test_label_encoder.shape))
    # Generate class weights
    logger.info("Generate class weights...")
    y_integers = np.argmax(train_label_encoder, axis=1)
    class_weights = class_weight.compute_class_weight("balanced",
                                                      np.unique(y_integers),
                                                      y_integers)
    class_weights_dic = dict(enumerate(class_weights))
    model_params.update({"class_weights": class_weights_dic})
    logger.info("class weight : {}".format(class_weights_dic))

    # Split train/valid/test
    logger.info("Split train/valid/test...")
    train_feature_split, validation_feature_split, train_split_label, validation_split_label = train_test_split(
        train_feature, train_label_encoder, test_size=0.2, random_state=0)
    logger.info(
        "train split feature shape      : {} train split label shape      : {}"
        .format(train_feature_split.shape, train_split_label.shape))
    logger.info(
        "validation split feature shape : {} validation split label shape : {}"
        .format(validation_feature_split.shape, validation_split_label.shape))

    logger.info("Train/Load model and predict...")
    cur_model = AmountSensitiveModel(
        train_feature_split, train_split_label, validation_feature_split,
        validation_split_label, test_feature, test_label_encoder,
        one_hot_encoder, model_params, class_weights, dic_path,
        params['train_motor'], params['test_motor'])
    if params["train_flag"]:
        cur_model.train_model()
    else:
        cur_model.predict_with_exist_model()

    # save params
    saved_params = {"params": params, "model_params": model_params}
    with open(dic_path + '/saved_params.json', 'w') as fp:
        json.dump(saved_params, fp)


if __name__ == "__main__":
    main()
