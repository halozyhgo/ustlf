# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import warnings

import lightgbm as lgb

from joblib import dump, load
from loguru import logger

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



class ML_Model:
    """
    初始化模型类的构造函数。

    参数:
    - model_name (str): 模型的名称。
    - model_params (dict): 模型的参数，通常是一个字典。
    - model_type (str): 模型的类型，例如'分类'、'回归'等。虽然在构造函数中未直接使用，但可能是子类或特定方法中需要的。
    - model_path (str): 模型保存的路径。

    属性:
    - model_name: 存储模型的名称。
    - model_params: 存储模型的参数。
    - model_path: 存储模型保存的路径。
    - model: 初始化为空，用于后续创建或加载模型实例。
    """
    def __init__(self, model_name, model_params, model_path):
        self.model_name = model_name    # 模型名称
        self.model_params = model_params    # 模型参数
        self.model_path = model_path        # 模型保存路径
        self.model = None                   # 初始化模型为空


    def save_model(self):
        """
        保存训练好的模型到指定路径。

        参数:
        model: 训练好的模型对象，可以来自任何机器学习库。

        此方法首先使用dump函数将模型保存到self.model_path指定的路径。
        然后记录一条信息，指示模型已成功保存。
        """
        if not os.path.exists(self.model_path):
            # print('测试一下是否创建文件')
            logger.info(f'创建{self.model_path}模型文件夹')
            # os.mkdir(self.model_path)
            os.makedirs(self.model_path)

        save_dir = self.model_path+'/'+self.model_name+'.pkl'
        dump(self.model, save_dir)
        logger.info(f'{self.model_name}模型保存成功')

    def load_model(self):
        """
        加载模型。

        本函数通过调用load函数来加载存储在self.model_path路径下的模型。load函数的具体实现细节
        在此省略，因为它涉及到外部依赖或框架的特定行为。简而言之，它负责从给定的路径中读取模型
        数据，并将其反序列化或解析为可使用的模型对象。

        返回:
            返回从model_path加载的模型对象。这个对象可以是机器学习模型、深度学习模型或其他类型的
            模型，具体取决于load函数的行为和model_path中的内容。
        """
        save_dir = self.model_path+'/'+self.model_name+'.pkl'
        self.model = load(save_dir)
        return load(save_dir)


    def get_model(self):
        """
        创建并返回一个LGBMRegressor模型实例。

        该方法根据实例中的`model_params`参数初始化一个轻量级梯度提升机（LightGBM）的回归器模型。
        这里使用`model_params`来配置模型属性，确保模型能够根据不同的需求进行定制。

        :return: 初始化后的LGBMRegressor模型实例。
        :rtype: lightgbm.LGBMRegressor
        """
        # 使用模型参数字典初始化LGBMRegressor模型
        self.model = lgb.LGBMRegressor(**self.model_params)
        # 返回初始化后的模型实例
        return self.model


    def model_train(self, X_train, y_train):
        """
        训练模型。

        该函数使用提供的训练数据集对模型进行训练。在训练开始和结束时，记录日志信息。

        参数:
        - X_train: 训练数据集的特征值。
        - y_train: 训练数据集的目标值。

        返回:
        - 训练完成的模型。
        """
        # 记录训练开始的日志信息
        # logger.info(f'开始训练{self.model_name}模型')

        # 使用训练数据集对模型进行训练
        self.model.fit(X_train, y_train)

        # 记录训练结束的日志信息
        # logger.info(f'{self.model_name}模型训练完毕')

        # 返回训练完成的模型
        return self.model

    def model_predict(self, X_features):
        """
        使用训练好的模型对输入的特征进行预测。

        参数:
        X_features: 二维数组，包含多条样本的特征数据。每行对应一个样本，每列对应一个特征。
                    也可以是单条数据，用于单步预测。

        返回:
        predict: 预测结果，一维数组。每个元素对应于输入样本的预测标签或回归值。
        """
        # 使用训练好的模型对输入的特征数据进行预测
        predict = self.model.predict(X_features)
        # 返回预测结果
        return predict
