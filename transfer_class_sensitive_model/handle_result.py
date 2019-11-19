import numpy as np
from sklearn import metrics

# path = 'saved_model/2019_07_20_15_28_33_cnn_lstm_sliding_20_motor_train_0_test_3'
# path = 'saved_model/2019_07_20_16_05_49_cnn_lstm_sliding_20_motor_train_1_test_3'
# path = 'saved_model/2019_07_20_16_27_14_cnn_lstm_sliding_20_motor_train_2_test_3'
# path = 'saved_model/2019_09_04_20_12_55_cnn_lstm_sliding_20_motor_train_0_test_0'
# path = "saved_model/2019_09_04_19_57_17_cnn_lstm_sliding_20_motor_train_3_test_3"
# path = "saved_model/2019_09_04_19_34_08_cnn_lstm_sliding_20_motor_train_2_test_2"
#path = "saved_model/2019_09_04_17_03_48_cnn_lstm_sliding_20_motor_train_1_test_1"
# path = "saved_model/2019_09_06_10_22_02_cnn_lstm_sliding_20_motor_train_0_test_0"

def generate_metrics(path, ifTrain=True):
        pre = 'test' if ifTrain else 'train'
        average_list = ['micro', 'macro', 'weighted', None]
        target_names = ['', 'class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9']


        test_y_score = np.load('{}/{}_predict_result.npy'.format(path, pre))
        test_y_pred = np.load('{}/{}_predict_result_inverse.npy'.format(path, pre))
        test_y_true = np.load('{}/{}_label.npy'.format(path, pre))
        test_y_true = np.reshape(test_y_true, (test_y_true.shape[0]))
        raw_test_y_true = np.load('{}/{}_label_encoder.npy'.format(path, pre))

        test_softmax_output = np.load('{}/{}_softmax_feature.npy'.format(path, pre))

        auc_score = [metrics.roc_auc_score(raw_test_y_true, test_y_score, average=ways) for ways in average_list]
        f1_score = [metrics.f1_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
        recall_score = [metrics.recall_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
        precision_score = [metrics.precision_score(test_y_true, test_y_pred, average=ways) for ways in average_list]
        confusion_matrix = metrics.confusion_matrix(test_y_true, test_y_pred)
        
        # logger.info("auc   micro : {:.6f}   macro : {:.6f}  weighted : {:.6f}",format(average_list[0], average_list[1], average_list[2]))
        # logger.info("f1_score :")
        # for i in range(len(average_list)):
        #     logger.info('{} : {} \n'.format(average_list[i], f1_score[i]))

        # logger.info("recall_score :")
        # for i in range(len(average_list)):
        #     logger.info('{} : {} \n'.format(average_list[i], recall_score[i]))

        # logger.info("precision_score :")
        # for i in range(len(average_list)):
        #     logger.info('{} : {} \n'.format(average_list[i], precision_score[i]))   
        
        # logger.info("confusion_matrix :")
        # for item in target_names:
        #     logger.info('{:<10s} '.format(item))
        # for i in range(len(confusion_matrix)):
        #     logger.info('{:<10s} '.format(target_names[i + 1])) 
        # for item in confusion_matrix[i]:
        #     logger.info('{:<10d} '.format(item))    
  
        
        with open(path + '/metrics_{}.txt'.format(pre),'w') as f:
            # auc
            f.write('auc : \n')
            [f.write('{} : {} \n'.format(average_list[i], auc_score[i])) for i in range(len(average_list))]
            f.write('\n')
            # f1_score
            f.write('f1_score : \n')
            [f.write('{} : {} \n'.format(average_list[i], f1_score[i])) for i in range(len(average_list))]
            f.write('\n')
            # recall_score
            f.write('recall_score : \n')
            [f.write('{} : {} \n'.format(average_list[i], recall_score[i])) for i in range(len(average_list))]
            f.write('\n')
            # precision_score
            f.write('precision_score : \n')
            [f.write('{} : {} \n'.format(average_list[i], precision_score[i])) for i in range(len(average_list))]
            f.write('\n')
            # confusion_matrix
            f.write('confusion_matrix : \n')
            for item in target_names:
                f.write('{:<10s} '.format(item)) 
            f.write('\n') 
            for i in range(len(confusion_matrix)):
                f.write('{:<10s} '.format(target_names[i + 1])) 
                for item in confusion_matrix[i]:
                    f.write('{:<10d} '.format(item)) 
                f.write('\n') 

# generate_metrics("saved_model/2019_10_29_15_46_16_cnn_lstm_sliding_20_motor_train_0_test_3")