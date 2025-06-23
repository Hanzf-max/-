import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import image_utils

def visualize_feature_maps(model, img_path):
    img = image_utils.load_img(img_path, target_size=(150, 150))
    img_tensor = image_utils.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)

    layer_names = [layer.name for layer in model.layers[:4]]
    images_pre_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_col = n_features // images_pre_row
        display_grid = np.zeros((size * n_col, images_pre_row * size))

        for col in range(n_col):
            for row in range(images_pre_row):
                channel_image = layer_activation[0, :, :, col * images_pre_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
    plt.show()
