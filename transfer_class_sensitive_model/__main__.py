import argparse
import json
import logging
import os
import time

import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from bearing.transfer_class_sensitive_model.model import \
    TransferClassSensitiveModel
from bearing.transfer_class_sensitive_model.tools import mkdir

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Transfer & imbalance model and modify the mmd compute \
            module based on the number of classed')
    parser.add_argument('--train-motor', type=str, default='0')
    parser.add_argument('--test-motor', type=str, default='3')
    parser.add_argument('--train-flag', type=str, default='True')
    parser.add_argument('--model-dic-path', type=str, default=None)

    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    parser.add_argument('--sliding-window-length', type=int, default=20)

    return parser.parse_args()


def main(args):
    module_path = os.path.dirname(os.path.abspath(__file__))
    pre_module_path = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))

    if args.train_flag:
        # mkdir
        dic_path = "{}/saved_model/{}_transfer_class_sensitive_model_{}_motor_train_{}_test_{}".format(
            module_path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
            args.sliding_window_length, args.train_motor, args.test_motor)
        mkdir(dic_path)
        logger.info("mkdir : " + dic_path)
    else:
        dic_path = args.model_dic_path
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
        format(pre_module_path, args.train_motor, args.sliding_window_length))
    train_label = np.load(
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy".
        format(pre_module_path, args.train_motor, args.sliding_window_length))

    test_feature = np.load(
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_feature_sample.npy".
        format(pre_module_path, args.test_motor, args.sliding_window_length))
    test_label = np.load(
        "{}/dataset/dataset_12k_motor_{}_sliding_window_{}_label_sample.npy".
        format(pre_module_path, args.test_motor, args.sliding_window_length))
    # test_label = np.load(
    #     "{}/Cnn_lstm_model/saved_model/2019_07_18_16_35_08_cnn_lstm_sliding_20_motor_train_0_test_3/test_predict_result_inverse.npy".format(pre_module_path)
    # )

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
    # model_params.update({"class_weights": class_weights_dic})
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

    data_dic = {
        'train_feature': train_feature_split,
        'train_label': train_split_label,
        'validation_feature': validation_feature_split,
        'validation_label': validation_split_label,
        'test_feature_for_transfer': test_feature[:len(train_feature_split)],
        'test_label_for_transfer': test_label_encoder[:len(train_split_label)],
        'validation_feature_for_transfer': validation_feature_split,
        'validation_label_for_transfer': validation_split_label,
        'test_feature': test_feature,
        'test_label': test_label_encoder
    }
    cur_model = TransferClassSensitiveModel(data_dic, one_hot_encoder, args,
                                            list(class_weights), dic_path)
    if args.train_flag:
        cur_model.train_model()
    else:
        cur_model.predict_with_exist_model()

    # save params
    saved_params = {"params": args}
    with open(dic_path + '/saved_params.json', 'w') as fp:
        json.dump(saved_params, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
