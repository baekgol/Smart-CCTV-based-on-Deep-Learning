import keras
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing import image
from keras.layers import *
from keras.models import Model
import keras.backend as K

import tensorflow as tf
from tensorflow.python.framework import ops

import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import deprocess_image

nb_classes = 3

def trained_model():
    base_model = Xception(input_shape=(img_width, img_height, 3), include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='sigmoid')(x)

    # add your top layer block to your base model
    model = Model(base_model.input, predictions)
    return model

def load_image(path, target_size=(299, 299)):
    x = image.load_img(path, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # 정규화

    return x

def generate_gradcam(img_tensor, model, class_index, activation_layer, graph):
    model_input = model.input

    # y_c : class_index에 해당하는 CNN 마지막 layer op(softmax, linear, ...)의 입력
    y_c = model.outputs[0].op.inputs[0][0, class_index]
    # y_c = model.outputs[0].op.inputs[0].op.inputs[0][0, class_index]

    # A_k: activation conv layer의 출력 feature map
    A_k = model.get_layer(activation_layer).output

    # model의 입력에 대해서,
    # activation conv layer의 출력(A_k)과
    # 최종 layer activation 입력(y_c)의 A_k에 대한 gradient,
    # 모델의 최종 출력(prediction) 계산
    with graph.as_default():
        get_output = K.function([model_input], [A_k, K.gradients(y_c, A_k)[0], model.output])
    [conv_output, grad_val, prediction] = get_output([img_tensor])

    # batch size가 포함되어 shape가 (1, width, height, k)이므로
    # (width, height, k)로 shape 변경
    # 여기서 width, height는 activation conv layer인 A_k feature map의 width와 height를 의미함
    conv_output = conv_output[0]
    grad_val = grad_val[0]

    # global average pooling 연산
    # gradient의 width, height에 대해 평균을 구해서(1/Z) weights(a^c_k) 계산
    weights = np.mean(grad_val, axis=(0, 1))

    # activation conv layer의 출력 feature map(conv_output)과
    # class_index에 해당하는 weights(a^c_k)를 k에 대응해서 weighted combination 계산

    # feature map(conv_output)의 (width, height)로 초기화
    grad_cam = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam += w * conv_output[:, :, k]

    # 계산된 weighted combination 에 ReLU 적용
    grad_cam = np.maximum(grad_cam, 0)

    return grad_cam, weights

def main(model, rawImage, rgbImage, graph, state):
    img_width = 224
    img_height = 224

    # img_tmp = img
    rgbImage = cv2.resize(rgbImage, dsize=(img_width, img_height))
    rgbImage = np.expand_dims(rgbImage, axis=0)
    rgbImage = preprocess_input(rgbImage)  # 정규화

    # predicted classes → 0 - Fire, 1 - Normal_state, 2 - Smoke, 3 - Fire-Smoke
    # 원본 이미지 predict
    # preds_org = model.predict(rgbImage)  # numpy array, shape(1, 3) - [[a, b, c]]
    # predicted_class_org = preds_org.argmax(axis=1)[0]  # 행별 최대값의 인덱스
    if state == 'Fire':
        predicted_class_org = 0
    # elif state == 'Smoke':
    #     predicted_class_org = 2

    conv_name = 'conv2d_4'  # 마지막 Convolution layer
    grad_cam_org, grad_val_org = generate_gradcam(rgbImage, model, predicted_class_org, conv_name, graph)

    # 이미지 출력
    rawImage = cv2.resize(rawImage, (img_width, img_height))  # 3차원
    grad_cam_org = cv2.resize(grad_cam_org, (img_width, img_height))    # 2차원

    grad_cam_org = (grad_cam_org - grad_cam_org.min()) / (grad_cam_org.max() - grad_cam_org.min())
    grad_cam_org = grad_cam_org * 255   # 역정규화
    grad_cam_org = grad_cam_org.astype(np.uint8)    # 값을 정수로 변경
    grad_cam_org = cv2.applyColorMap(grad_cam_org, cv2.COLORMAP_JET)    # RGB 채널로 변경
    grad_cam_org = cv2.addWeighted(rawImage, 0.5, grad_cam_org, 0.5, 0.0)

    return grad_cam_org

if __name__ == "__main__":
    img_width = 224
    img_height = 224

    model = trained_model()

    model.load_weights("model/loss_weights_190714(b16_e200).h5")
    img_path = '../DataSet/Multilabeled_Classification/test/Fire/fire-smoke2(none)_frame_0.JPG'
    img = load_image(path=img_path, target_size=(img_width, img_height))  # numpy array, shape(1, 224, 224, 3), 정규화된 이미지
    print(img)

    # 4등분 분할 이미지
    imgPart = [0 for _ in range(4)]
    imgPart[0] = img[0:1, 0:int(img_height/2), 0:int(img_width/2)]
    imgPart[1] = img[0:1, 0:int(img_height/2), int(img_width/2):img_width]
    imgPart[2] = img[0:1, int(img_height/2):img_height, 0:int(img_width/2)]
    imgPart[3] = img[0:1, int(img_height/2):img_height, int(img_width/2):img_width]

    # predicted classes → 0 - Fire, 1 - Normal_state, 2 - Smoke, 3 - Fire-Smoke
    # 원본 이미지 predict
    preds_org = model.predict(img)  # numpy array, shape(1, 3) - [[a, b, c]]
    predicted_class_org = preds_org.argmax(axis=1)[0]  # 행별 최대값의 인덱스
    print("predicted classes(org):", preds_org)
    print("predicted top1 class(org):", predicted_class_org)

    # 분할 이미지 predict
    preds = [0 for _ in range(4)]
    predicted_class = [0 for _ in range(4)]
    for i in range(4):
        imgPart[i] = np.squeeze(imgPart[i], axis=0) # 차원 축소
        imgPart[i] = cv2.resize(imgPart[i], (img_width, img_height))    # 모델 사이즈 맞추기
        imgPart[i] = np.expand_dims(imgPart[i], axis=0) # 차원 추가

        preds[i] = model.predict(imgPart[i])
        predicted_class[i] = preds[i].argmax(axis=1)[0]
        print("predicted classes(imgPart[%d])" %(i) + ":", preds[i])
        print("predicted top1 class(imgPart[%d])" %(i) + ":", predicted_class[i])

    conv_name = 'conv2d_4'  # 마지막 Convolution layer
    # grad_cam - numpy array, shape(10, 10), grad_val - numpy array, shape(1024,)
    grad_cam = [0 for _ in range(4)]
    grad_val = [0 for _ in range(4)]
    for i in range(4):
        grad_cam[i], grad_val[i] = generate_gradcam(imgPart[i], model, predicted_class[i], conv_name)
    grad_cam_org, grad_val_org = generate_gradcam(img, model, predicted_class_org, conv_name)

    # 이미지 출력
    img = cv2.imread(img_path)  # 비정규화된 이미지
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    grad_cam_org = cv2.resize(grad_cam_org, (img_width, img_height))

    imgPart[0] = img[0:int(img_height / 2), 0:int(img_width / 2)]
    imgPart[1] = img[0:int(img_height / 2), int(img_width / 2):img_width]
    imgPart[2] = img[int(img_height / 2):img_height, 0:int(img_width / 2)]
    imgPart[3] = img[int(img_height / 2):img_height, int(img_width / 2):img_width]
    for i in range(4):
        grad_cam[i] = cv2.resize(grad_cam[i], (int(img_width / 2), int(img_height / 2)))

    # Original
    plt.figure(0)
    plt.imshow(img)
    plt.axis('off')

    # Original(Grad-Cam)
    plt.figure(1)
    plt.imshow(img)
    plt.imshow(grad_cam_org, cmap="jet", alpha=.5)  # max - red, min - blue
    plt.colorbar()
    plt.axis('off')

    # imgPart[0]
    plt.figure(2)
    plt.imshow(imgPart[0])
    plt.imshow(grad_cam[0], cmap="jet", alpha=.5)  # max - red, min - blue
    plt.axis('off')

    # imgPart[1]
    plt.figure(3)
    plt.imshow(imgPart[1])
    plt.imshow(grad_cam[1], cmap="jet", alpha=.5)  # max - red, min - blue
    plt.axis('off')

    # imgPart[2]
    plt.figure(4)
    plt.imshow(imgPart[2])
    plt.imshow(grad_cam[2], cmap="jet", alpha=.5)  # max - red, min - blue
    plt.axis('off')

    # imgPart[3]
    plt.figure(5)
    plt.imshow(imgPart[3])
    plt.imshow(grad_cam[3], cmap="jet", alpha=.5)  # max - red, min - blue
    plt.axis('off')

    plt.show()