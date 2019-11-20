import gc
import logging
import os
import sys
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import (LSTM, BatchNormalization, Conv1D, Dense, Dropout,
                          Flatten, Input, Lambda, concatenate)
from keras.metrics import categorical_accuracy
from keras.models import Model, Sequential, load_model
from keras.preprocessing import sequence
from keras.regularizers import Regularizer
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.signal import savgol_filter
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf

parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)
from constants import const
import json
import handle_result
import plot_lstm_feature

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--train-motor', type=str, default = '0')
parser.add_argument('--train-flag', type=str, default = 'True')
parser.add_argument('--model-dic-path', type=str, default = None)
args = parser.parse_args()


def mkdir(path):

    folder = os.path.exists(path)

    if not folder:  #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  #makedirs 创建文件时如果路径不存在会创建这个路径


def get_random_block_from_test(data, size):
    sample = np.random.randint(len(data), size=size)
    data = data[sample]
    return data


class CnnLstmModel():
    def __init__(self, train_feature, train_label, validation_feature,
                 validation_label, test_feature, test_label, one_hot_encoder,
                 model_params, dic_path, train_motor, test_motor):
        self.train_feature = train_feature
        self.train_label = train_label
        self.validation_feature = validation_feature
        self.validation_label = validation_label
        self.test_feature = test_feature
        self.test_label = test_label
        self.one_hot_encoder = one_hot_encoder
        self.model_params = model_params
        self.dic_path = dic_path
        self.train_motor = train_motor
        self.test_motor = test_motor

        self.show_eval_figure = False
        self.model = None
        self.history = None

    def _model(self):
        input_train = Input(
            shape=(const.SLIDING_WINDOW_LENGTH, 2), name="input_train")

        conv_1_train = Conv1D(
            10, 3, strides=1, activation="tanh", name="conv_1_shared")(input_train)

        conv_1_dropout = Dropout(0.3)(conv_1_train)

        lstm_1 = LSTM(
            self.model_params["hidden_size"] , return_sequences=True,
            activation="tanh")(conv_1_dropout)
        lstm_1_dropout = Dropout(0.3)(lstm_1)

        flatten = Flatten()(lstm_1_dropout)
        output = Dense(10, activation="softmax")(flatten)

        self.model = Model(inputs=input_train, outputs=output)

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"])

    def _save_model(self):
        logger.info("Saved model...")
        self.model.save(
            self.dic_path + "/model.h5")  # creates a HDF5 file "my_model.h5"

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
        train_label_inverse = np.reshape(
            train_label_inverse,
            (train_label_inverse.shape[0]))
        np.save(self.dic_path + "/train_label.npy",
                train_label_inverse)

        logger.info("Get softmax layer output...")
        get_softmax_layer_output = K.function([self.model.layers[0].input], [
            self.model.layers[-1].output
        ])
        train_softmax_layer_output = get_softmax_layer_output([self.train_feature])
        test_softmax_layer_output = get_softmax_layer_output([self.test_feature])

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

        # cnn feature
        # np.save(self.dic_path + "/train_cnn_feature.npy",
        #         train_layer_output[0])
        # np.save(self.dic_path + "/test_cnn_feature.npy", test_layer_output[0])

        # lstm fature
        np.save(self.dic_path + "/train_lstm_feature.npy",
                train_layer_output[1])
        np.save(self.dic_path + "/test_lstm_feature.npy", test_layer_output[1])

        # softmax output
        np.save(self.dic_path + "/train_softmax_feature.npy",
                train_layer_output[2])
        np.save(self.dic_path + "/test_softmax_feature.npy",
                test_layer_output[2])
        

    def train_model(self):
        self._model()
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=self.model_params["early_stopping_patience"], verbose=1)
        self.history = self.model.fit(
            x=self.train_feature,
            y=self.train_label,
            epochs=self.model_params["epochs"],
            verbose=self.model_params["verbose"],
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
        handle_result.generate_metrics(self.dic_path)

    def predict_with_exist_model(self):
        self._load_exist_model()
        self._get_predict_result_and_middle_feature()
        handle_result.generate_metrics(self.dic_path)
        # plot_lstm_feature.plot(self.dic_path, self.train_motor, self.test_motor)

    def get_model(self):
        return self.model


def main():
    params = {
        "train_motor": args.train_motor,
        "test_motor": 3,

        "train_flag": False,
        # "model_dic_path": "saved_model/2019_07_20_16_27_14_cnn_lstm_sliding_20_motor_train_2_test_3",
        # model_path = "saved_model/2019_07_20_15_28_33_cnn_lstm_sliding_20_motor_train_0_test_3/model.h5"
        # model_path = "saved_model/2019_07_20_16_05_49_cnn_lstm_sliding_20_motor_train_1_test_3/model.h5"     
        # "model_dic_path": "saved_model/2019_09_04_20_36_14_cnn_lstm_sliding_20_motor_train_0_test_0"  
        # "model_dic_path": "saved_model/2019_09_04_22_09_31_cnn_lstm_sliding_20_motor_train_3_test_3"  
        # "model_dic_path": "saved_model/2019_09_04_20_49_52_cnn_lstm_sliding_20_motor_train_1_test_1" 
        # "model_dic_path": "saved_model/2019_09_04_21_07_13_cnn_lstm_sliding_20_motor_train_2_test_2" 
        # "model_dic_path": "saved_model/2019_07_18_16_46_38_cnn_lstm_sliding_20_motor_train_1_test_3"    
        "model_dic_path": args.model_dic_path                                                                                                                                                                                                                                      
    }
    model_params = {
        "batch_size": 512,
        "hidden_size": 32,

        "epochs": 100,
        "verbose": 0,
        "shuffle": True,
        "early_stopping_patience": 5,
    }

    if params["train_flag"]:
        # mkdir
        dic_path = "saved_model/{}_cnn_lstm_sliding_{}_motor_train_{}_test_{}".format(
            time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
            const.SLIDING_WINDOW_LENGTH, params["train_motor"], params["test_motor"])
        mkdir(dic_path)
        logger.info("mkdir : " + dic_path)
    else:
        dic_path = params["model_dic_path"]
        logger.info("exit model dir : " + dic_path)

    logging.basicConfig(
        filename=dic_path + "/" +
        str(os.path.basename(__file__).split(".")[0] + ".log"),
        # stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # load data
    logger.info("Loading feature and label...")
    train_feature = np.load(
        "../dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy".
        format(params["train_motor"], const.SLIDING_WINDOW_LENGTH))
    train_label = np.load(
        "../dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy".
        format(params["train_motor"], const.SLIDING_WINDOW_LENGTH))

    test_feature = np.load(
        "../dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy".
        format(params["test_motor"], const.SLIDING_WINDOW_LENGTH))
    test_label = np.load(
        "../dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy".
        format(params["test_motor"], const.SLIDING_WINDOW_LENGTH))
    
    # train_feature, test_feature, train_label, test_label = train_test_split(
    #     train_feature, train_label, test_size=0.2, random_state=0)

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
    cur_model = CnnLstmModel(
        train_feature_split, train_split_label, validation_feature_split,
        validation_split_label, test_feature, test_label_encoder,
        one_hot_encoder, model_params, dic_path, params['train_motor'], params['test_motor'])
    if params["train_flag"]:
        cur_model.train_model()
    else:
        cur_model.predict_with_exist_model()

    # save params
    saved_params = {
        "params": params,
        "model_params": model_params
    }
    with open(dic_path + '/saved_params.json', 'w') as fp:
        json.dump(saved_params, fp)


if __name__ == "__main__":
    main()