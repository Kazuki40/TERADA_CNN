
# パッケージ
import os
import csv
from tabnanny import verbose
from tkinter import N, TRUE
from xml.etree.ElementTree import TreeBuilder
import cv2
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    GlobalAveragePooling2D,
    MaxPool2D,
    Add,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.applications import (
    VGG16,
    ResNet50,
    DenseNet121,
    EfficientNetB0,
    MobileNetV3Small,
    NASNetMobile,
)
import csv
import pandas as pd

# データ
# ラベルのリスト
class_label = [
    "Bridge",
    "Building",
    "Castle",
    "Ground",
    "Nature_Mountain",
    "Nature_Waterfron",
    "Road",
]


# データの前処理
def image_func(iadge_pass):
    images_data = []
    for i in range(len(iadge_pass)):
        path = os.path.abspath(iadge_pass[i])
        buf = np.fromfile(path, np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        images_data.append(image)

    images_data = np.array(images_data)
    return images_data


# データ読み込み部


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
train_images = image_func(tran_image_pass)

# 検証用
test_images = []
test_images = image_func(test_image_pass)


# アーリーストップ
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)


######################################

# データ拡張
data = pd.read_csv("data_image.csv", index_col=0)
data_ans = []
np.random.seed(0)  # 乱数シードを314に設定

for data_temp_loop_i in range(1, 26):
    print(data_temp_loop_i)
    # ImageDataGeneratorのインスタンスを作成
    # ttps://pynote.hatenablog.com/entry/keras-image-data-generator
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=data["rotation_range"][data_temp_loop_i],
        width_shift_range=data["shift_range"][data_temp_loop_i],
        height_shift_range=data["shift_range"][data_temp_loop_i],
        brightness_range=(
            (1 - data["brightness_range"][data_temp_loop_i]),
            (1 + data["brightness_range"][data_temp_loop_i]),
        ),
        shear_range=data["shear_range"][data_temp_loop_i],
        zoom_range=data["zoom_range"][data_temp_loop_i],
        channel_shift_range=data["channel_shift_range"][data_temp_loop_i],
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
    )

    # VGG16
    # VGG16モデルの読み込み
    np.random.seed(0)  # 乱数シードを314に設定
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

    # VGG16モデルに層を追加
    x = vgg16.output
    x = Flatten()(x)
    x = Dense(4096)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(len(class_label))(x)
    x = Activation("softmax")(x)

    model = Model(inputs=vgg16.input, outputs=x)

    # モデルのコンパイル
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    np.random.seed(0)  # 乱数シードを314に設定

    # モデルの学習

    history = model.fit(
        datagen.flow(train_images, train_data_label, batch_size=32),
        validation_data=(test_images, test_data_label),
        epochs=500,
        verbose=2,
        shuffle=True,
        callbacks=[early_stopping],
    )

    # モデル評価
    test_loss, test_accuracy = model.evaluate(test_images, test_data_label)
    data_ans.extend([test_loss])
    
        # ファイルを開く（追記モード）
    with open("output.txt", "a") as f:
        # データを追記する
        print(test_loss, file=f)



data["ans"] = pd.Series(data_ans, name="ans")