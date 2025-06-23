import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import image_utils
from sklearn.model_selection import train_test_split
import os
from keras.models import Sequential, load_model
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers, activations
from pathlib import Path
import matplotlib.pyplot as plt


# 读取Baked Potato，Burger,Fries,Pizza,Sandwich,Taquito快餐的JPEG格式文件名
dir_BakedPotato = Path(r'archive\Fast Food Classification V2\Train/Baked Potato')
filepaths_BakedPotato = list(dir_BakedPotato.glob('**/*.JPEG'))

dir_Burger = Path(r'archive\Fast Food Classification V2\Train/Burger')
filepaths_Burger = list(dir_Burger.glob('**/*.JPEG'))

dir_Fries = Path(r'archive\Fast Food Classification V2\Train/Fries')
filepaths_Fries = list(dir_Fries.glob('**/*.JPEG'))

# 构建列表，存储图片路径
filepaths = [filepaths_BakedPotato,filepaths_Burger,filepaths_Fries,]
# 把二维列表降一维
filepaths = sum(filepaths, [])

# 构建列表，存储图片对应类型
list_BakedPotato = list(map(lambda x:'BakedPotato', [i for i in range(len(filepaths_BakedPotato))]))
list_Burger = list(map(lambda x:'Burger', [i for i in range(len(filepaths_Burger))]))
list_Fries = list(map(lambda x:'Fries', [i for i in range(len(filepaths_Fries))]))
label = [list_BakedPotato,list_Burger,list_Fries,]
print(len(label))

# 把二维列表降一维
label = sum(label, [])
print(len(label))
# 合并两个列表为数据框
filepaths_S =  pd.Series(filepaths,name='FilePaths')
label_S = pd.Series(label,name='labels')
data = pd.merge(filepaths_S,label_S,right_index=True,left_index=True)
print(data.head())
print(data.info())

# 查看图像
pic = plt.figure(figsize=(12,7))
l1 = [3,1502,3002]
for i in range(1,4,1):
    ax = pic.add_subplot(2, 3, i)
    plt.imshow(plt.imread(data['FilePaths'][l1[i-1]]))
    plt.title(data['labels'][l1[i-1]])
plt.savefig('图像.jpg')
plt.show()

# 划分数据集,85:15划分x_train,x_test
data['FilePaths'] = data['FilePaths'].astype(str)
X_train, X_test = train_test_split(data, test_size=0.15,stratify=data['labels'])
print('训练集形状', X_train.shape)
print('测试集形状', X_test.shape)

# 划分数据集,4:1划分x_train,x_val
X_train, X_val = train_test_split(X_train, test_size=0.2,stratify=X_train['labels'])
print('训练集形状', X_train.shape)
print('验证集形状', X_val.shape)

# 查看各类型的图片张数
print(X_train['labels'].value_counts())
print(X_train['FilePaths'].shape)

# 图像预处理
img_preprocessing = ImageDataGenerator(rescale=1./255)
x_train = img_preprocessing.flow_from_dataframe(dataframe=X_train,
                                                x_col='FilePaths',
                                                y_col='labels',
                                                target_size=(112, 112),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=32,
                                                seed=30)

x_test = img_preprocessing.flow_from_dataframe(dataframe=X_test,
                                               x_col='FilePaths',
                                               y_col='labels',
                                               target_size=(112, 112),
                                               color_mode='rgb',
                                               class_mode='categorical',
                                               batch_size=32,
                                               seed=30)

x_val = img_preprocessing.flow_from_dataframe(dataframe=X_val,
                                              x_col='FilePaths',
                                              y_col='labels',
                                              target_size=(112, 112),
                                              color_mode='rgb',
                                              class_mode='categorical',
                                              batch_size=32,
                                              seed=30)

# 构建神经网络模型并训练
model = Sequential()

# Conv2D层，32个滤波器
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

# Conv2D层，64个滤波器
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

# Conv2D层，128个滤波器
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))

# 展平数据，降维
model.add(Flatten())

# 全连接层
model.add(Dense(256))
model.add(Activation('relu'))

# 减少过拟合
model.add(Dropout(0.5))

# 全连接层
model.add(Dense(3)) # 识别6种类
model.add(Activation('softmax')) # #使用softmax进行分类


# 模型编译
model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4), loss="categorical_crossentropy",
              metrics=["accuracy"]) # metrics指定衡量模型的指标



# 加载h1模型
h1 = load_model('h21')

# h1模型预测
# 3类型
l1 = ['BakedPotato','Burger','Fries',]
# 原图
i = 10
p1 = plt.figure(figsize=(12,7))
img1,label1 = x_test.next()
plt.rcParams["font.sans-serif"]=["FangSong"] # 解决中文显示异常
ax1 = p1.add_subplot(2, 1, 1)
plt.imshow((img1[i]*255).astype('uint8'))
plt.title('实际类型为：' + l1[np.argmax(label1[i])])

# 将图片转换成 4D 张量
x_test_img1 = img1[i].reshape(1, 112, 112, 3).astype("float32")

# 预测结果的概率柱状图
pr1 = h1.predict(x_test_img1)
ax2 = p1.add_subplot(2, 1, 2)
plt.title("预测结果的概率柱状图")
plt.bar(np.arange(3), pr1.reshape(3), align="center")
plt.xticks(np.arange(3),l1)
plt.show()

# 实际类型与h1模型预测结果对比
print('实际类型：' + l1[np.argmax(label1[i])])
print('h1模型预测的结果是：' + l1[np.argmax(pr1)])
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 获取真实标签和预测标签
y_true = []
y_pred = []

# 获取类名和标签映射（如 {'BakedPotato': 0, 'Burger': 1, 'Fries': 2}）
class_indices = x_test.class_indices
labels = list(class_indices.keys())

# 遍历整个测试集
for i in range(len(x_test)):
    x_batch, y_batch = x_test[i]
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred_batch = h1.predict(x_batch)
    y_pred.extend(np.argmax(y_pred_batch, axis=1))

# 生成混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of h1 Model on Test Set')
plt.savefig('h1混淆矩阵.jpg')
plt.show()

# 加载h2模型
h2 = load_model('h22')

# h2模型预测
# 3类型
l2 = ['BakedPotato','Burger','Fries']
# 原图
i = 10
p2 = plt.figure(figsize=(12,7))
img2,label2 = x_test.next()
plt.rcParams["font.sans-serif"]=["FangSong"] # 解决中文显示异常
ax3 = p2.add_subplot(2, 1, 1)
plt.imshow((img2[i]*255).astype('uint8'))
plt.title('实际类型为：' + l2[np.argmax(label2[i])])

# 将图片转换成 4D 张量
x_test_img2 = img2[i].reshape(1, 112, 112, 3).astype("float32")

# 预测结果的概率柱状图
pr2 = h2.predict(x_test_img2)
ax4 = p2.add_subplot(2, 1, 2)
plt.title("预测结果的概率柱状图")
plt.bar(np.arange(3), pr2.reshape(3), align="center")
plt.xticks(np.arange(3),l2)
plt.show()

# 实际类型与h2模型预测结果对比
print('实际类型：' + l2[np.argmax(label2[i])])
print('h2模型预测的结果是：' + l2[np.argmax(pr2)])
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 获取真实标签和预测标签
y_true = []
y_pred = []

# 获取类名和标签映射（如 {'BakedPotato': 0, 'Burger': 1, 'Fries': 2}）
class_indices = x_test.class_indices
labels = list(class_indices.keys())

# 遍历整个测试集
for i in range(len(x_test)):
    x_batch, y_batch = x_test[i]
    y_true.extend(np.argmax(y_batch, axis=1))
    y_pred_batch = h2.predict(x_batch)
    y_pred.extend(np.argmax(y_pred_batch, axis=1))

# 生成混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of h2 Model on Test Set')
plt.savefig('h2混淆矩阵.jpg')
plt.show()
