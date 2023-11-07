import os
import csv
from tabnanny import verbose
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class_label = [
    "Bridge",
    "Building",
    "Castle",
    "Ground",
    "Nature_Mountain",
    "Nature_Waterfron",
    "Road",
]

def get_gradcam(model, img_path, class_index, layer_name):
    path = os.path.abspath(img_path)
    buf = np.fromfile(path, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)    
    img = cv2.resize(image, (224, 224))

    # 前処理用の画像を作成
    preprocessed_img = img / 255.0
    preprocessed_img = np.array(preprocessed_img)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(preprocessed_img)

    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(layer_name)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(preprocessed_img)
        class_out = model_out[:, class_index]

    grads = tape.gradient(class_out, last_conv_layer)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    heatmap = np.uint8(255 * heatmap)

    if len(heatmap.shape) == 3 and heatmap.shape[0] == 1:
        heatmap = np.squeeze(heatmap, axis=0)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    superimposed_img = cv2.addWeighted(img, 0.2, heatmap, 0.8, 0)

    return superimposed_img


#モデル呼び出し
#model = load_model("terada_vgg16.h5")

# Grad-CAMを生成する画像のパスとクラスインデックスを指定
#img_path ="data/Bridge/IMG_1982.jpg" # 画像のパスを指定

#class_index = 0  # クラスのインデックスを指定
#layer_name = "block5_conv3"  # Grad-CAMを計算する対象の層を指定

# Grad-CAMを生成
#gradcam = get_gradcam(model, img_path, class_index, layer_name)

# Grad-CAMを表示
#plt.imshow(gradcam)
#plt.show()


#モデル呼び出し
loaded_model = load_model("terada_vgg16_7.keras")

def online_estimate(pic_pass):



    #画像処理
    path = os.path.abspath(pic_pass)
    buf = np.fromfile(path, np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
   
    image = np.array(image)
    img_batch = np.expand_dims(image, axis=0)#4次元化
    

    #データの予測
    predictions = loaded_model.predict(img_batch)
    class_index = np.argmax(predictions)  # 最も確率の高いクラスのインデックス
    predicted_class = list(class_label)[class_index]  # インデックスをクラス名に変換
    probability = predictions[0][class_index]  # 最も確率の高いクラスの確率
    print(f"estimate_class: {predicted_class}")
    print(f"provibility: {probability}")   


    layer_name="block5_conv3"
    
    a=get_gradcam(loaded_model, pic_pass, class_index, layer_name)
    
    # Grad-CAMを表示
    plt.imshow(a)
    plt.show()
    plt.pause(10)  # 10秒間待機
    

kumamoto_pass = "data/Nature_Waterfron/P2180488.jpg"
online_estimate(kumamoto_pass)
