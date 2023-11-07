import os
import csv
from tabnanny import verbose
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

# tf.nondifferentiable_batch_function()

print("preparation")
import numpy as np
np.random.seed(0)  # 乱数シードを314に設定


class_label = [
    "Bridge",
    "Building",
    "Castle",
    "Ground",
    "Nature_Mountain",
    "Nature_Waterfron",
    "Road",
]

tran_image_pass = []  # pass
train_data_label = []  # ラベル
test_image_pass = []  # pass
test_data_label = []  # ラベル


# データがあるファイルを読み込む
for i, now_class_name in enumerate(class_label):
    # txtファイルを読み込む
    txt_file_name = "data/" + now_class_name + "/file_name.txt"

    # ファイルを開く
    with open(txt_file_name, "r") as f:
        # ファイルの各行を読み込む
        temp_txt_file = f.read().splitlines()

    # ラベル
    data_image_pass = []
    data_label_temp = []

    for j, temp_file in enumerate(temp_txt_file):
        # pass
        data_image_pass.append("data/" + now_class_name + "/" + temp_file)
        data_label_temp.append(now_class_name)

    # データを8：2に分ける
    (
        class_train_images_pass,
        class_val_images_pass,
        class_train_labels,
        class_val_labels,
    ) = train_test_split(
        data_image_pass, data_label_temp, test_size=0.2, random_state=0
    )

    # 答えの記入
    tran_image_pass.extend(class_train_images_pass)
    train_data_label.extend(class_train_labels)
    test_image_pass.extend(class_val_images_pass)
    test_data_label.extend(class_val_labels)


# ラベルの変更
label_mapping = {label: index for index, label in enumerate(class_label)}
train_data_label_num = [label_mapping[label] for label in train_data_label]
train_data_label = to_categorical(train_data_label_num, num_classes=len(class_label))
test_data_label_num = [label_mapping[label] for label in test_data_label]
test_data_label = to_categorical(test_data_label_num, num_classes=len(class_label))


# 画像データの読み込み

# 学習用
train_images = []
for i in range(len(tran_image_pass)):
    path = os.path.abspath(tran_image_pass[i])
    buf = np.fromfile(path, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    train_images.append(image)

train_images = np.array(train_images)

# 検証用
test_images = []
for i in range(len(test_image_pass)):
    path = os.path.abspath(test_image_pass[i])
    buf = np.fromfile(path, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    test_images.append(image)

test_images = np.array(test_images)

# ImageDataGeneratorのインスタンスを作成
# ttps://pynote.hatenablog.com/entry/keras-image-data-generator
datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    brightness_range=(
        (1),
        (1),
    ),
    shear_range=0,
    zoom_range=0,
    channel_shift_range=0,
    fill_mode="nearest",
    cval=0,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=100,
    min_delta=0,
    verbose=1,
    mode="auto",
    restore_best_weights=True,
)

# loaded_model = load_model("terada_vgg16.h5")


# VGG16
# VGG16モデルの読み込み
vgg16 = VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling="max",
    classes=None,
)

# 転移学習
for layer in vgg16.layers:
    layer.trainable = False


# vgg16 = VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(224,224,3), pooling=None, classes=len(class_label))
# model=Model(inputs=vgg16.input, outputs=vgg16.output)
# VGG16モデルに層を追加
x = vgg16.output
x = Flatten()(x)
x = Dense(4096)(x)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.5)(x)
x = Dense(4096)(x)
# x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dense(len(class_label))(x)
x = Activation("softmax")(x)

model = Model(inputs=vgg16.input, outputs=x)



# モデルのコンパイル
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# モデルの学習
history = model.fit(
    train_images,
    train_data_label,
    batch_size=32,
    validation_data=(test_images, test_data_label),
    epochs=1024,
    verbose=2,
    shuffle=True,
    callbacks=[early_stopping],
)

# history = model.fit(
#     datagen.flow(train_images, train_data_label, batch_size=32),
#     validation_data=(test_images, test_data_label),
#     epochs=1024,
#     verbose=1,
#     shuffle=True,
#     callbacks=[early_stopping],
# )




# 学習パラメータの保存
model.save("terada_vgg16_7.keras")
# loaded_model = load_model("terada_vgg16.h5")

print("ans")


def show_history(history):
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0, 0].set_title("loss")
    ax[0, 0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0, 1].set_title("test_loss")
    ax[0, 1].plot(history.epoch, history.history["val_loss"], label="Validation loss")

    ax[1, 0].set_title("categorical_accuracy")
    ax[1, 0].plot(history.epoch, history.history["accuracy"], label="Train accuracy")
    ax[1, 1].set_title("test_accuracy")
    ax[1, 1].plot(
        history.epoch, history.history["val_accuracy"], label="Validation accuracy"
    )

    ax[0, 0].legend()
    ax[1, 0].legend()
    ax[0, 1].legend()
    ax[1, 1].legend()
    plt.show()


# モデル評価
test_loss, test_accuracy = model.evaluate(test_images, test_data_label)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# モデルの予測
predictions = model.predict(test_images)

# 混合行列
# モデルの予測結果をクラスインデックスに変換
predicted_labels = np.argmax(predictions, axis=1)

# 真のラベルをクラスインデックスに変換
true_labels = test_data_label_num

# 混同行列の計算
confusion = confusion_matrix(true_labels, predicted_labels)

# 混同行列をCSVファイルに保存
confusion_csv = "confusion_matrixa_7.csv"
with open(confusion_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(confusion)

# 混同行列を出力
print("Confusion Matrix:")
print(confusion)

# 分類レポートを出力
classification_rep = classification_report(
    true_labels, predicted_labels, target_names=class_label, output_dict=True
)
classification_csv = "classification_reporta_7.csv"

with open(classification_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
    for class_name, metrics in classification_rep.items():
        if class_name == "accuracy":
            continue
        writer.writerow(
            [
                class_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"],
            ]
        )

print("Classification Report:")
print(classification_rep)


# 予測結果を出力
# 予測結果をCSVファイルに書き込む
output_file = "ans_7.csv"

with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Sample", "Predicted Class", "True Class", "Probability", "Pass"])

    for i in range(len(predictions)):
        class_index = np.argmax(predictions[i])  # 最も確率の高いクラスのインデックス
        predicted_class = list(class_label)[class_index]  # インデックスをクラス名に変換
        true_class = list(class_label)[test_data_label_num[i]]  # 真のクラス
        probability = predictions[i][class_index]  # 最も確率の高いクラスの確率
        pass_n = test_image_pass[i]

        # 答え
        writer.writerow(
            [
                i + 1,
                predicted_class,
                true_class,
                probability,
                pass_n,
            ]
        )


show_history(history)