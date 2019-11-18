import json
import logging
import os
import time
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn import preprocessing

# parent_path = os.path.dirname(sys.path[0])
# if parent_path not in sys.path:
#     sys.path.append(parent_path)

from bearing.constants import const
from bearing.transfer_class_sensitive_model.model import TransferClassSensitiveModel
from bearing.transfer_class_sensitive_model.tools import mkdir

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    module_path = os.path.dirname(os.path.abspath(__file__))
    pre_module_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = {
        "train_motor": 1,
        "test_motor":
        3,
        "train_flag":
        True,
        # "model_dic_path": "saved_model/2019_07_20_16_27_14_cnn_lstm_sliding_20_motor_train_2_test_3",
        # model_path = "saved_model/2019_07_20_15_28_33_cnn_lstm_sliding_20_motor_train_0_test_3/model.h5"
        # model_path = "saved_model/2019_07_20_16_05_49_cnn_lstm_sliding_20_motor_train_1_test_3/model.h5"
        # "model_dic_path": "saved_model/2019_09_04_20_12_55_cnn_lstm_sliding_20_motor_train_0_test_0",
        # "model_dic_path": "saved_model/2019_09_04_19_57_17_cnn_lstm_sliding_20_motor_train_3_test_3",
        # "model_dic_path": "saved_model/2019_09_04_19_34_08_cnn_lstm_sliding_20_motor_train_2_test_2",
        "model_dic_path":
        "{}/saved_model/2019_11_18_16_59_02_cnn_lstm_sliding_20_motor_train_0_test_3".format(module_path),
        # "saved_model/2019_09_27_14_40_48_cnn_lstm_sliding_20_motor_train_0_test_3",
    }
    model_params = {
        "batch_size": 512,
        "hidden_size": 32,
        "epochs": 100,
        "verbose": 1,
        "shuffle": True,
        "early_stopping_patience": 5,
    }

    if params["train_flag"]:
        # mkdir
        dic_path = "{}/saved_model/{}_cnn_lstm_sliding_{}_motor_train_{}_test_{}".format(module_path,
            time.strftime("%Y_%m_%d_%H_%M_%S",
                          time.localtime()), const.SLIDING_WINDOW_LENGTH,
            params["train_motor"], params["test_motor"])
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
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy".
        format(pre_module_path, params["train_motor"], const.SLIDING_WINDOW_LENGTH))
    train_label = np.load(
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy".
        format(pre_module_path, params["train_motor"], const.SLIDING_WINDOW_LENGTH))

    train_feature, test_feature, train_label, test_label = train_test_split(
        train_feature, train_label, test_size=0.2, random_state=0)

    test_feature = np.load(
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy".
        format(pre_module_path, params["test_motor"], const.SLIDING_WINDOW_LENGTH))
    test_label = np.load(
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy".
        format(pre_module_path, params["test_motor"], const.SLIDING_WINDOW_LENGTH))

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
    cur_model = TransferClassSensitiveModel(
        train_feature_split, train_split_label, validation_feature_split,
        validation_split_label, test_feature[:len(train_feature_split)],
        test_label_encoder[:len(train_split_label)], validation_feature_split,
        validation_split_label,
        test_feature, test_label_encoder, one_hot_encoder, model_params,
        list(class_weights), dic_path, params['train_motor'],
        params['test_motor'])
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
