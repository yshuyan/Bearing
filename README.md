## 背景

工业数据的**分布差异**和**非均衡特性**影响了故障诊断与预测模型的准确性与泛化性。本项目旨在利用领域分布自适应算法和代价敏感非均衡算法来解决工业场景的实际问题。
[CWRU轴承数据集]: https://csegroups.case.edu/bearingdatacenter/home
## 环境

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
git clone 
```

解压至本地

### 目录树
.
├── amount_sensitive_model   
│   ├── amount_sensitive_model.py  
│   ├── handle_result.py  
│   ├── __init__.py  
│   ├── nohup.out  
│   └── __pycache__  
│       ├── amount_sensitive_model.cpython-37.pyc  
│       ├── handle_result.cpython-37.pyc   
│       └── __init__.cpython-37.pyc  
├── cnn_lstm_model  
│   ├── cnn_lstm_model.py  
│   ├── handle_result.py  
│   ├── __init__.py  
│   ├── nohup.out  
│   └── __pycache__  
│       ├── cnn_lstm_model.cpython-37.pyc  
│       ├── handle_result.cpython-37.pyc  
│       └── __init__.cpython-37.pyc  
├── constants.py  
├── .gitignore  
├── __init__.py  
├── __init__.pyc  
├── plot_lstm_feature.py  
├── __pycache__  
│   ├── constants.cpython-37.pyc  
│   ├── __init__.cpython-37.pyc  
│   └── plot_lstm_feature.cpython-37.pyc  
├── README.md  
├── transfer_class_sensitive_model  
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
├── transfer_iter_model  
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
├── transfer_model  
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
├── t-sne  
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
