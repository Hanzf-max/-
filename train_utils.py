from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt

def get_generators(X_train, X_val, X_test, augment=False):
    if augment:
        train_gen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')
    else:
        train_gen = ImageDataGenerator(rescale=1./255)

    val_gen = ImageDataGenerator(rescale=1./255)

    x_train = train_gen.flow_from_dataframe(dataframe=X_train,
                                            x_col='FilePaths',
                                            y_col='labels',
                                            target_size=(112, 112),
                                            class_mode='categorical',
                                            batch_size=32,
                                            seed=30)

    x_val = val_gen.flow_from_dataframe(dataframe=X_val,
                                        x_col='FilePaths',
                                        y_col='labels',
                                        target_size=(112, 112),
                                        class_mode='categorical',
                                        batch_size=32,
                                        seed=30)

    x_test = val_gen.flow_from_dataframe(dataframe=X_test,
                                         x_col='FilePaths',
                                         y_col='labels',
                                         target_size=(112, 112),
                                         class_mode='categorical',
                                         batch_size=32,
                                         seed=30)

    return x_train, x_val, x_test

def compile_and_train(model, x_train, x_val, epochs=80):
    model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, validation_data=x_val, epochs=epochs)
    return history

def plot_metrics(history, prefix='h1'):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plt.plot(range(1, len(loss)+1), loss, "bo-", label="Training Loss")
    plt.plot(range(1, len(val_loss)+1), val_loss, "ro--", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(f'{prefix}模型损失变化曲线图.jpg')
    plt.clf()

    plt.plot(range(1, len(acc)+1), acc, "bo-", label="Training Acc")
    plt.plot(range(1, len(val_acc)+1), val_acc, "ro--", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.savefig(f'{prefix}模型准确率变化曲线图.jpg')
    plt.clf()
