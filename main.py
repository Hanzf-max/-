import tkinter as tk
from tkinter import ttk
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from data_loader import load_data
from train_utils import get_generators

# 加载数据和模型
X_train, X_val, X_test = load_data()
_, _, x_test = get_generators(X_train, X_val, X_test)
img_batch, label_batch = x_test.next()
h1 = load_model("h21")
h2 = load_model("h22")
labels = ['BakedPotato', 'Burger', 'Fries']

# GUI 回调函数
def predict_and_display(model_name, img_index):
    img = img_batch[img_index]
    label = label_batch[img_index]
    input_img = img.reshape(1, 112, 112, 3).astype("float32")

    model = h1 if model_name == "h1" else h2
    pred = model.predict(input_img)[0]

    actual_class = labels[np.argmax(label)]
    predicted_class = labels[np.argmax(pred)]

    # 图像显示
    img_show = (img * 255).astype(np.uint8)
    image = Image.fromarray(img_show).resize((224, 224))  # 加大图片
    img_tk = ImageTk.PhotoImage(image)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    actual_label.config(text=f"实际类型：{actual_class}", font=("FangSong", 14))
    predicted_label.config(text=f"预测类型：{predicted_class}", font=("FangSong", 14))

    # 更新预测概率分布图
    ax.clear()
    ax.bar(np.arange(len(labels)), pred, color='skyblue')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("Predicted Probability", fontsize=14)
    fig.tight_layout()
    canvas.draw()
    canvas.flush_events()

# 创建窗口
root = tk.Tk()
root.title("快餐图像分类模型比较")
root.geometry("550x800")  # 加大窗口尺寸

# 顶部控制区
tk.Label(root, text="选择模型：", font=("FangSong", 12)).grid(row=0, column=0, padx=10, pady=5, sticky='e')
model_var = tk.StringVar(value="h1")
model_select = ttk.Combobox(root, textvariable=model_var, values=["h1", "h2"], font=("FangSong", 12), width=10)
model_select.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="图像索引（0-31）：", font=("FangSong", 12)).grid(row=1, column=0, padx=10, pady=5, sticky='e')
index_var = tk.IntVar(value=0)
index_entry = tk.Entry(root, textvariable=index_var, font=("FangSong", 12), width=10)
index_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Button(root, text="开始预测", font=("FangSong", 12), command=lambda: predict_and_display(model_var.get(), index_var.get())).grid(row=2, column=0, columnspan=2, pady=10)

# 图像显示
image_label = tk.Label(root)
image_label.grid(row=3, column=0, columnspan=2, pady=10)

# 文字标签
actual_label = tk.Label(root, text="实际类型：", font=("FangSong", 14))
actual_label.grid(row=4, column=0, columnspan=2, pady=5)

predicted_label = tk.Label(root, text="预测类型：", font=("FangSong", 14))
predicted_label.grid(row=5, column=0, columnspan=2, pady=5)

# 图表显示
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
fig, ax = plt.subplots(figsize=(5, 3))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=2, pady=10)

# 启动主循环
root.mainloop()
