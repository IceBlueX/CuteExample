# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:07:50 2018
@author: anshuai1
"""
'''
完整的利用embeddding处理类别特征的程序。
'''

"""
    使用 embedding 对语句进行降维转换
    经过 one_hot 即 pd.get_dummies() 之后的数据就是高纬度的稀疏数据
    实现高纬度稀疏向量的到低纬度稠密向量的特征降维
    原文链接
    https://blog.csdn.net/anshuai_aw1/article/details/83586404
"""

import numpy as np

import tensorflow as tf
import random as rn

# random seeds for stochastic parts of neural network
np.random.seed(10)
tf.random.set_seed(15)

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding

# ===================================================================================================
# 保证神经网络结果的复现
# ===================================================================================================
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.random.set_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
# ===================================================================================================

# 记录类别特征embedding后的维度。key为类别特征索引，value为embedding后的维度
cate_embedding_dimension = {'0': 3, '1': 2}


def build_embedding_network():
    # 以网络结构embeddding层在前，dense层在后。即训练集的X必须以分类特征在前，连续特征在后。
    inputs = []
    embeddings = []

    input_cate_feature_1 = Input(shape=(1,))
    embedding = Embedding(10, 3, input_length=1)(input_cate_feature_1)
    # embedding后是10*1*3，为了后续计算方便，因此使用Reshape转为10*3
    embedding = Reshape(target_shape=(3,))(embedding)
    inputs.append(input_cate_feature_1)
    embeddings.append(embedding)

    input_cate_feature_2 = Input(shape=(1,))
    embedding = Embedding(4, 2, input_length=1)(input_cate_feature_2)
    embedding = Reshape(target_shape=(2,))(embedding)
    inputs.append(input_cate_feature_2)
    embeddings.append(embedding)

    input_numeric = Input(shape=(1,))
    embedding_numeric = Dense(16)(input_numeric)
    inputs.append(input_numeric)
    embeddings.append(embedding_numeric)

    x = Concatenate()(embeddings)

    x = Dense(10, activation='relu')(x)
    x = Dropout(.15)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)

    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


# ===================================================================================================
# 程序入口
# ===================================================================================================
'''
输入数据是32*3，32个样本，2个类别特征，1个连续特征。
对类别特征做entity embedding，第一个类别特征的可能值是0到9之间（10个），第二个类别特征的可能值是0到3之间（4个）。
对这2个特征做one-hot的话，应该为32*14，
对第一个类别特征做embedding使其为3维，对第二个类别特征做embedding使其为2维。3维和2维的设定是根据实验效果和交叉验证设定。
对连续特征不做处理。
这样理想输出的结果就应该是32*6，其中，类别特征维度为5，连续特征维度为1。
'''
# ===================================================================================================
# 构造训练数据
# ===================================================================================================

sample_num = 32  # 样本数为32
cate_feature_num = 2  # 类别特征为2
contious_feature_num = 1  # 连续特征为1

# 保证了训练集的复现
rng = np.random.RandomState(123)
cate_feature_1 = rng.randint(10, size=(32, 1))
cate_feature_2 = rng.randint(4, size=(32, 1))
contious_feature = rng.rand(32, 1)
X = []
X.append(cate_feature_1)
X.append(cate_feature_2)
X.append(contious_feature)
# 二分类
Y = np.random.rand(32, 1)

# ===================================================================================================
# 训练和预测
# ===================================================================================================
# train
NN = build_embedding_network()
NN.fit(X, Y, epochs=3, batch_size=4, verbose=0)
# predict
y_preds = NN.predict(X)[:, 0]

# 画出模型，需要GraphViz包。
# from keras.utils import plot_model
# plot_model(NN, to_file='NN.png')

# ===================================================================================================
# 读embedding层的输出结果
# ===================================================================================================

model = NN  # 创建原始模型
for i in range(cate_feature_num):
    # 由NN.png图可知，如果把类别特征放前，连续特征放后，cate_feature_num+i就是所有embedding层
    layer_name = NN.get_config()['layers'][cate_feature_num + i]['name']

    intermediate_layer_model = Model(inputs=NN.input,
                                     outputs=model.get_layer(layer_name).output)

    # numpy.array
    intermediate_output = intermediate_layer_model.predict(X)

    intermediate_output.resize([32, cate_embedding_dimension[str(i)]])

    if i == 0:
        X_embedding_trans = intermediate_output
    else:
        X_embedding_trans = np.hstack((X_embedding_trans, intermediate_output))  # 水平拼接

# 取出原来的连续特征。这里的list我转numpy一直出问题，被迫这么写循环了。
for i in range(contious_feature_num):
    if i == 0:
        X_contious = X[cate_feature_num + i]
    else:
        X_contious = np.hstack((X_contious, X[cate_feature_num + i]))

# ===================================================================================================
# 在类别特征做embedding后的基础上，拼接连续特征，形成最终矩阵，也就是其它学习器的输入
# ===================================================================================================

'''
最终的结果：32*6.其中，类别特征维度为5（前5个），连续特征维度为1（最后1个）
'''
X_trans = np.hstack((X_embedding_trans, X_contious))

'''
好了，我们现在来验证一下embeddding后的结果是不是一个索引的结果表。
以第一个类别特征为例，利用代码NN.trainable_weights[0].eval(session=sess)我们输出embedding_1层的参数。
-0.0464945	0.0284733	-0.0365357
0.051283	0.0336468	0.0440866
0.0370058	-0.0378573	-0.0357488
0.0249379	0.031956	0.024898
-0.0075664	0.0355627	-0.0149643
-0.0481578	-0.0210528	0.0118361
-0.0178293	-0.0212218	0.0246742
0.0160812	0.0294887	-0.0069619
0.0200302	0.0472979	0.0312307
0.0416624	0.0408308	0.0405323
这是一个10*3的矩阵，符合我们的想法，即应该one-hot的10维变为3维。
为了方便，我们只看第一个类别特征cate_feature_1的前5行：
2
2
6
1
3
去索引权重，结果应该是：
0.0370058	-0.0378573	-0.0357488
0.0370058	-0.0378573	-0.0357488
-0.0178293	-0.0212218	0.0246742
0.051283	0.0336468	0.0440866
0.0249379	0.031956	0.024898
我们接着去查看X_trans中的前5行前3列：
发现确实是这样的结果。
我们又一次证明了embedding层的输出就是类别特征的值索引权重矩阵的结果！
值得注意的是：embedding层权重矩阵的训练跟其它层没有什么区别，都是反向传播更新的。
'''





