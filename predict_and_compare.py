import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def predict_and_show(model_path, x_test, labels, index=10):
    model = load_model(model_path)
    img_batch, label_batch = x_test.next()

    plt.rcParams["font.sans-serif"]=["FangSong"]
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imshow((img_batch[index] * 255).astype("uint8"))
    plt.title("实际类型为：" + labels[np.argmax(label_batch[index])])

    x_img = img_batch[index].reshape(1, 112, 112, 3).astype("float32")
    pr = model.predict(x_img)

    ax2 = fig.add_subplot(2, 1, 2)
    plt.bar(np.arange(len(labels)), pr.reshape(len(labels)), align="center")
    plt.xticks(np.arange(len(labels)), labels)
    plt.title("预测结果的概率柱状图")
    plt.show()

    print("实际类型：" + labels[np.argmax(label_batch[index])])
    print("模型预测结果：" + labels[np.argmax(pr)])
