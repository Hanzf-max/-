import os
from keras.models import load_model
from data_loader import load_data
from model_builder import build_model
from train_utils import get_generators, compile_and_train, plot_metrics
from predict_and_compare import predict_and_show
from visualize_filters import visualize_feature_maps

# 切换模式
TRAIN_MODE = False  # 设置为 True 时重新训练，False 时只做推理

X_train, X_val, X_test = load_data()
x_train, x_val, x_test = get_generators(X_train, X_val, X_test)

if TRAIN_MODE:
    model = build_model()
    history = compile_and_train(model, x_train, x_val)
    model.save("h21")
    plot_metrics(history, prefix='h1')

    x_train_aug, x_val_aug, _ = get_generators(X_train, X_val, X_test, augment=True)
    history2 = compile_and_train(model, x_train_aug, x_val_aug)
    model.save("h22")
    plot_metrics(history2, prefix='h2')
else:
    model = load_model("h22")


img_path = "archive\Fast Food Classification V2\Train/Burger/Burger-Train (1470).jpeg"
import numpy as np
from keras.utils import image_utils
img = image_utils.load_img(img_path, target_size=(150,150))
img_tensor = image_utils.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()
# 可视化特征图（只对一个模型执行即可）
visualize_filters = True
if visualize_filters:
    visualize_feature_maps(model, "archive/Fast Food Classification V2/Train/Burger/Burger-Train (1470).jpeg")

