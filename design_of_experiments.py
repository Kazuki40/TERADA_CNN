
# �p�b�P�[�W
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

# �f�[�^
# ���x���̃��X�g
class_label = [
    "Bridge",
    "Building",
    "Castle",
    "Ground",
    "Nature_Mountain",
    "Nature_Waterfron",
    "Road",
]


# �f�[�^�̑O����
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


# �f�[�^�ǂݍ��ݕ�


tran_image_pass = []  # pass
train_data_label = []  # ���x��
test_image_pass = []  # pass
test_data_label = []  # ���x��


# �f�[�^������t�@�C����ǂݍ���
for i, now_class_name in enumerate(class_label):
    # txt�t�@�C����ǂݍ���
    txt_file_name = "data/" + now_class_name + "/file_name.txt"

    # �t�@�C�����J��
    with open(txt_file_name, "r") as f:
        # �t�@�C���̊e�s��ǂݍ���
        temp_txt_file = f.read().splitlines()

    # ���x��
    data_image_pass = []
    data_label_temp = []

    for j, temp_file in enumerate(temp_txt_file):
        # pass
        data_image_pass.append("data/" + now_class_name + "/" + temp_file)
        data_label_temp.append(now_class_name)

    # �f�[�^��8�F2�ɕ�����
    (
        class_train_images_pass,
        class_val_images_pass,
        class_train_labels,
        class_val_labels,
    ) = train_test_split(
        data_image_pass, data_label_temp, test_size=0.2, random_state=0
    )

    # �����̋L��
    tran_image_pass.extend(class_train_images_pass)
    train_data_label.extend(class_train_labels)
    test_image_pass.extend(class_val_images_pass)
    test_data_label.extend(class_val_labels)


# ���x���̕ύX
label_mapping = {label: index for index, label in enumerate(class_label)}
train_data_label_num = [label_mapping[label] for label in train_data_label]
train_data_label = to_categorical(train_data_label_num, num_classes=len(class_label))
test_data_label_num = [label_mapping[label] for label in test_data_label]
test_data_label = to_categorical(test_data_label_num, num_classes=len(class_label))


# �摜�f�[�^�̓ǂݍ���

# �w�K�p
train_images = []
train_images = image_func(tran_image_pass)

# ���ؗp
test_images = []
test_images = image_func(test_image_pass)


# �A�[���[�X�g�b�v
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    min_delta=0,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)


######################################

# �f�[�^�g��
data = pd.read_csv("data_image.csv", index_col=0)
data_ans = []
np.random.seed(0)  # �����V�[�h��314�ɐݒ�

for data_temp_loop_i in range(1, 26):
    print(data_temp_loop_i)
    # ImageDataGenerator�̃C���X�^���X���쐬
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
    # VGG16���f���̓ǂݍ���
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�
    vgg16 = VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=None,
    )

    # �]�ڊw�K
    for layer in vgg16.layers:
        layer.trainable = False

    # VGG16���f���ɑw��ǉ�
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

    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    # ���f���̊w�K

    history = model.fit(
        datagen.flow(train_images, train_data_label, batch_size=32),
        validation_data=(test_images, test_data_label),
        epochs=500,
        verbose=2,
        shuffle=True,
        callbacks=[early_stopping],
    )

    # ���f���]��
    test_loss, test_accuracy = model.evaluate(test_images, test_data_label)
    data_ans.extend([test_loss])
    
        # �t�@�C�����J���i�ǋL���[�h�j
    with open("output.txt", "a") as f:
        # �f�[�^��ǋL����
        print(test_loss, file=f)



data["ans"] = pd.Series(data_ans, name="ans")