## 背景

工业数据的**分布差异**和**非均衡特性**影响了故障诊断与预测模型的准确性与泛化性。本项目旨在利用领域分布自适应算法和代价敏感非均衡算法来解决工业场景的实际问题。

提出了一种基于深度领域自适应的非均衡故障诊断模型（Imbalanced Fault Diagnostics Model based on Domain Adaptation, IFDM-DA）。该模型利用代价敏感算法调整了损失函数中不同类别的权重，提高了分类器对故障样本的侧重程度。利用非监督的方法获得了测试集中样本的伪标签，并使用领域自适应中的条件分布自适应算法，计算了训练集与测试集中相同类别样本的特征差异。将该差异大小作为损失函数的一项，在训练过程中不断缩小训练集与测试集相同类别特征的差异，提升了模型的泛化性以及少数类（样本数量较少的故障类）的分类准确率。最后在凯斯西储大学（Case Western Reserve University, CWRU）轴承数据集上对整体实验的多项指标进行评估

## 模型结构

![](https://raw.githubusercontent.com/yshuyan/Picture/master/img/t.png)

## 软件环境

框架：Keras（TensorFlow后端）

语言：Python（3.7.3）

配置：GPU（12G）

相关包版本：



Package                            Version  
---------------------------------- ---------
conda                              4.7.12   
conda-build                        3.18.8   
conda-package-handling             1.3.11  
conda-verify                       3.4.2   
flake8                             3.7.9   
h5py                               2.9.0   
imageio                            2.5.0   
jupyter                            1.0.0   
jupyterlab                         1.0.2   
Keras                              2.2.4   
Keras-Applications                 1.0.8   
Keras-Preprocessing                1.1.0   
numpy                              1.16.4  
numpydoc                           0.9.1   
pandas                             0.24.2   
pep8                               1.7.1    
pip                                19.1.1   
pylint                             2.3.1   
scikit-image                       0.15.0   
scikit-learn                       0.21.2   
scipy                              1.3.0   
seaborn                            0.9.0   
tensorboard                        1.14.0   
tensorflow                         1.14.0   
tensorflow-estimator               1.14.0   

## 代码

### 安装

```
git clone https://github.com/yshuyan/Bearing.git
```

解压至本地

### 目录树

.
├── amount_sensitive_model   # 基准模型 + 非均衡算法
│   ├── amount_sensitive_model.py  
│   ├── handle_result.py  
│   ├── __init__.py  
│   ├── nohup.out  
│   └── __pycache__  
│       ├── amount_sensitive_model.cpython-37.pyc  
│       ├── handle_result.cpython-37.pyc   
│       └── __init__.cpython-37.pyc  
├── cnn_lstm_model  # 基准模型
│   ├── cnn_lstm_model.py  
│   ├── handle_result.py  
│   ├── __init__.py  
│   ├── nohup.out  
│   └── __pycache__  
│       ├── cnn_lstm_model.cpython-37.pyc  
│       ├── handle_result.cpython-37.pyc  
│       └── __init__.cpython-37.pyc  
├── constants.py  # 常变量定义
├── .gitignore  
├── __init__.py  
├── __init__.pyc  
├── plot_lstm_feature.py  # tsne 绘图
├── __pycache__  
│   ├── constants.cpython-37.pyc  
│   ├── __init__.cpython-37.pyc  
│   └── plot_lstm_feature.cpython-37.pyc  
├── README.md  
├── transfer_class_sensitive_model  # 基准模型 + 非均衡 + 迁移（非迭代）
│   ├── handle_result.py  
│   ├── __init__.py  
│   ├── __init__.pyc  
│   ├── __main__.py  
│   ├── model.py  
│   ├── nohup.out  
│   ├── __pycache__  
│   │   ├── handle_result.cpython-37.pyc  
│   │   ├── __init__.cpython-37.pyc  
│   │   ├── __main__.cpython-37.pyc  
│   │   ├── model.cpython-37.pyc  
│   │   └── tools.cpython-37.pyc  
│   └── tools.py  
├── transfer_iter_model  # 基准模型 + 非均衡 + 迁移（迭代）
│   ├── data_generator.py  
│   ├── dataset_loader.py  
│   ├── handle_result.py  
│   ├── __init__.py  
│   ├── __main__.py  
│   ├── model.py  
│   ├── nohup.out  
│   ├── __pycache__  
│   │   ├── data_generator.cpython-37.pyc  
│   │   ├── handle_result.cpython-37.pyc  
│   │   ├── __init__.cpython-37.pyc  
│   │   ├── __main__.cpython-37.pyc  
│   │   ├── model.cpython-37.pyc  
│   │   └── tools.cpython-37.pyc  
│   └── tools.py  
├── transfer_model  # 测试
│   ├── handle_result copy.py  
│   ├── handle_result.py  
│   ├── __main__.py  
│   ├── model.py  
│   ├── nohup copy.out  
│   ├── nohup.out  
│   ├── __pycache__  
│   │   └── handle_result.cpython-37.pyc  
│   ├── tools.py  
│   └── transfer_model.py  
├── t-sne  # tsne 工具
│   ├── __init__.py  
│   └── plot_lstm_feature.py  
└── .vscode  
    └── settings.json  

## 调用

```shell
	# 基准模型调用
    python -m bearing.cnn_lstm_model --train-motor=0 --flag
    python -m bearing.cnn_lstm_model --train-motor=1 --flag
    python -m bearing.cnn_lstm_model --train-motor=2 --flag

	# 基准模型调用（--no-flag可调用已有模型）
    python -m bearing.cnn_lstm_model --train-motor=0 --no-flag --model-dic-path=saved_model/2019_10_29_15_02_55_cnn_lstm_sliding_20_motor_train_0_test_3
	
	# 基准模型 + 非均衡模型
    python -m bearing.amount_sensitive_model.amount_sensitive_model --flag --train-motor=0

    # 基准模型 + 非均衡模型 + 迁移学习（非迭代）
    python -m bearing.transfer_class_sensitive_model --train-motor=0 --flag 
    
    # 基准模型 + 非均衡模型 + 迁移学习（迭代）
    python -m bearing.transfer_iter_model --train-motor=0 --flag 
```

### Tips：

--train-motor：训练集负载

--flag：训练模式

--no-flag：调用已有模型，与--model-dic-path同时选用

--model-dic-path：已有模型地址

## 结果

A/D实验四种模型各项指标总体对比

|              | AUC      | F1      | 召回率  | 精确率  |
| ------------ | -------- | ------- | ------- | ------- |
| CNN-LSTM     | 0.94548  | 0.61280 | 0.63342 | 0.64868 |
| CNN-LSTM(CS) | 0.949666 | 0.62376 | 0.63673 | 0.69048 |
| IFDM-DA(NI)  | 0.96772  | 0.63065 | 0.60377 | 0.71378 |
| IFDM-DA      | 0.96488  | 0.60112 | 0.63770 | 0.67665 |

B/D实验四种模型各项指标总体对比

|                | AUC     | F1      | 召回率  | 精确率  |
| -------------- | ------- | ------- | ------- | ------- |
| CNN-LSTM       | 0.91693 | 0.44437 | 0.51844 | 0.51252 |
| CNN-LSTM（CS） | 0.95610 | 0.59294 | 0.63206 | 0.64278 |
| IFDM-DA（NI）  | 0.96300 | 0.61324 | 0.62385 | 0.61124 |
| IFDM-DA        | 0.96675 | 0.61048 | 0.60127 | 0.62674 |

C/D实验四种模型各项指标总体对比

|                | AUC     | F1      | 召回率  | 精确率  |
| -------------- | ------- | ------- | ------- | ------- |
| CNN-LSTM       | 0.97742 | 0.75073 | 0.73524 | 0.77806 |
| CNN-LSTM（CS） | 0.98241 | 0.67227 | 0.66750 | 0.88959 |
| IFDM-DA（NI）  | 0.98834 | 0.73741 | 0.77833 | 0.75035 |
| IFDM-DA        | 0.99030 | 0.81726 | 0.77221 | 0.89612 |

迁移效果：

> 迁移前

![](https://raw.githubusercontent.com/yshuyan/Picture/master/img/20200622230409.png)

> 迁移后

![](https://raw.githubusercontent.com/yshuyan/Picture/master/img/20200622230527.png)
