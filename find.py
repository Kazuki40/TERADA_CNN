
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

# tf.nondifferentiable_batch_function()
print("preparation")

np.random.seed(0)  # �����V�[�h��314�ɐݒ�


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

def show_history(history, model_name, make_file_name):
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

    now_plit1 = "terada_plot_" + model_name + "_" + str(make_file_name) + ".jpg"
    plt.savefig(now_plit1)


def train_and_evaluate_model(
    model,
    train_images,
    train_data_label,
    test_images,
    test_data_label,
    early_stopping,
    class_label,
    test_data_label_num,
    model_name,
    make_file_name,
):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�
    if make_file_name == 0:
        # ���f���̊w�K
        history = model.fit(
            train_images,
            train_data_label,
            batch_size=32,
            validation_data=(test_images, test_data_label),
            epochs=500,
            verbose=2,
            shuffle=True,
            callbacks=[early_stopping],
        )
    else:
        history = model.fit(
            datagen.flow(train_images, train_data_label, batch_size=32),
            validation_data=(test_images, test_data_label),
            epochs=500,
            verbose=2,
            shuffle=True,
            callbacks=[early_stopping],
        )

    save_name = "terada_" + model_name + "_" + str(make_file_name) + ".keras"
    # �w�K�p�����[�^�̕ۑ�
    model.save(save_name)

    # ���f���]��
    test_loss, test_accuracy = model.evaluate(test_images, test_data_label)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # ���f���̗\��
    predictions = model.predict(test_images)

    # �����s��
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_data_label_num
    confusion = confusion_matrix(true_labels, predicted_labels)

    # �����s���CSV�t�@�C���ɕۑ�
    confusion_csv = "confusion_matrixa_" + model_name + "_" + str(make_file_name) + ".csv"
    with open(confusion_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(confusion)

    # �����s����o��
    print("Confusion Matrix:")
    print(confusion)

    # ���ރ��|�[�g���o��
    classification_rep = classification_report(
        true_labels, predicted_labels, target_names=class_label, output_dict=True
    )

    classification_csv = (
        "classification_reporta_" + model_name + "_" + str(make_file_name) + ".csv"
    )

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

    # �\�����ʂ��o��
    output_file = "ans_" + model_name + "_" + str(make_file_name) + ".csv"
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Sample", "Predicted Class", "True Class", "Probability", "Pass"]
        )
        for i in range(len(predictions)):
            class_index = np.argmax(predictions[i])
            predicted_class = list(class_label)[class_index]
            true_class = list(class_label)[test_data_label_num[i]]
            probability = predictions[i][class_index]
            pass_n = test_image_pass[i]
            writer.writerow(
                [
                    i + 1,
                    predicted_class,
                    true_class,
                    probability,
                    pass_n,
                ]
            )

    show_history(history, model_name, make_file_name)


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

######################################

# �f�[�^�g��

# ImageDataGenerator�̃C���X�^���X���쐬
# ttps://pynote.hatenablog.com/entry/keras-image-data-generator
datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=45.0,
    width_shift_range=1.0,
    height_shift_range=1.0,
    brightness_range=(0.5,1.5), 
    shear_range=1.0,
    zoom_range=0.5,
    channel_shift_range=0.0,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
)


# �A�[���[�X�g�b�v
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=50,
    min_delta=0,
    verbose=0,
    mode="auto",
    restore_best_weights=True,
)

# loaded_model = load_model("terada_vgg16.h5")


# ���f��0
for j in range(2):
    # VGG16
    # VGG16���f���̓ǂݍ���
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�
    model_name="GPT3"
    
    model = Sequential()
    model.add(Conv2D(32,(3,3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_label), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )
    
    

# ���f��
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "NASNetMobile"

    NAS = NASNetMobile(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
    model = Model(inputs=NAS.input, outputs=NAS.output)

    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )





# ���f��7
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "ResNet50_transfer"

    resnet50 = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
        # �]�ڊw�K
    for layer in resnet50.layers:
        layer.trainable = False
        
    x = resnet50.output
    x = Dense(len(class_label))(x)
    x = Activation("softmax")(x)


    model = Model(inputs=resnet50.input, outputs=x)

    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )



# ���f��8
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "ResNet50"

    resnet50 = ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
  
    model = Model(inputs=resnet50.input, outputs=resnet50.output)

    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )


# ���f��9
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "DenseNet121_transfer"

    densenet121 = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
    

        # �]�ڊw�K
    for layer in densenet121.layers:
        layer.trainable = False
        
    x = densenet121.output
    x = Dense(len(class_label))(x)
    x = Activation("softmax")(x)
  
    model = Model(inputs=densenet121.input, outputs=x)


    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )





# ���f��10
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "DenseNet121"

    densenet121 = DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
  
    model = Model(inputs=densenet121.input, outputs=densenet121.output)


    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )

# ���f��9
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "EfficientNetB0_transfer"

    efficientnetb0 = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
    

        # �]�ڊw�K
    for layer in efficientnetb0.layers:
        layer.trainable = False
        
    x = efficientnetb0.output
    x = Dense(len(class_label))(x)
    x = Activation("softmax")(x)
  
    model = Model(inputs=efficientnetb0
                  
                  .input, outputs=x)


    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )



# ���f��7
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "EfficientNetB0"
  
    efficientnetb0 = EfficientNetB0(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
  
    model = Model(inputs=efficientnetb0.input, outputs=efficientnetb0.output)



    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )


# ���f��8
for j in range(2):
    np.random.seed(0)  # �����V�[�h��314�ɐݒ�

    model_name = "MobileNetV3Small"

    Mnet = MobileNetV3Small(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling="max",
        classes=len(class_label),
    )
  
    model = Model(inputs=Mnet.input, outputs=Mnet.output)


    # ���f���̃R���p�C��
    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )


    # ���f��5
for j in range(2):
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

    model_name = "vgg16_transfer_7_g"

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
    model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )

# ���f��6
for j in range(2):
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
        layer.trainable = True

    model_name = "vgg16_7_g"

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
    model.summary()

    make_data_name = j

    train_and_evaluate_model(
        model,
        train_images,
        train_data_label,
        test_images,
        test_data_label,
        early_stopping,
        class_label,
        test_data_label_num,
        model_name,
        make_data_name,
    )
    