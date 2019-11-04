import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as k
import matplotlib.pyplot as plt

# fix seed for reproducible results (only works on CPU, not GPU)
from utils import ImageDataGenerator

seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 3  # number of classes
img_width, img_height = 224, 224  # change based on the shape/structure of your images
batch_size = 8  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 200  # number of iteration the algorithm gets trained.

def train(train_data_dir, validation_data_dir, model_path):
    base_model = Xception(input_shape=(img_width, img_height, 3), include_top=False)

    x = base_model.output

    # 필터를 기준으로 나누어 평균값만 뽑아냄
    x = GlobalAveragePooling2D()(x)

    # 완전연결계층을 만들어줌
    # sigmoid - True(0.5~1), False(0~0.5), 선형을 비선형으로 변경, output layer에서 사용
    # Dense() - hidden layer의 Node 수 정의
    # multi label일 경우 sigmoid, 아닐 경우 softmax
    predictions = layers.Dense(nb_classes, activation='sigmoid')(x)

    # 모델 만들기(input, output)
    model = Model(base_model.input, predictions)
    print(model.summary())

    for layer in model.layers:
        layer.trainable = True  # layer 학습 가능하게 설정
        layer.kernel_regularizer = l2(0.05) # 가중치 규제(L2 정규화)

    # binary_crossentropy - 0 or 1로 판단
    # crossentropy - 하나만 1, 나머지 0으로 판단
    model.compile(optimizer='nadam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 학습 데이터 가공
    # multi label일 경우 multi_categorical, 아닐 경우 categorical
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       horizontal_flip=True,
                                       zoom_range=[0.8,4.5],
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='multi_categorical')
    # ImageDataGenerator 이미지 확인
    # import matplotlib.pyplot as plt
    # for item in train_generator:
    #     X, y = item
    #     print(X.shape)
    #     plt.figure()
    #     plt.imshow(X[0])
    #     plt.show()


    print(train_generator.class_indices)

    # 검증 데이터 가공
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                                  target_size=(img_width, img_height),
                                                                  batch_size=batch_size,
                                                                  class_mode='multi_categorical')

    # 가중치 저장 경로
    final_acc_weights_path = os.path.join(os.path.abspath(model_path), 'acc_weights_190920(b8_e200_r0.05).h5')
    final_loss_weights_path = os.path.join(os.path.abspath(model_path), 'loss_weights_190920(b8_e200_r0.05).h5')

    # 콜백 설정
    callbacks_list = [
        ModelCheckpoint(final_acc_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        ModelCheckpoint(final_loss_weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    ]

    # ImageDataGenerator 사용 시 fit_generator()로 학습
    hist = model.fit_generator(train_generator,
                               epochs=nb_epoch,
                               validation_data=validation_generator,
                               callbacks=callbacks_list,
                               steps_per_epoch=1000)

    # 모델 저장
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)

    # 결과 그래프 출력
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='Train - Loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='Val - Loss')

    acc_ax.plot(hist.history['acc'], 'b', label='Train - Accuracy')
    acc_ax.plot(hist.history['val_acc'], 'g', label='Val - Accuracy')

    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    acc_ax.set_ylabel('Accuracy')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

if __name__ == '__main__':
    # 데이터셋 경로 지정
    data_dir = '../DataSet/Multilabeled_Classification'
    train_dir = os.path.join(os.path.abspath(data_dir), 'train')  # Inside, each class should have it's own folder
    validation_dir = os.path.join(os.path.abspath(data_dir), 'val')  # each class should have it's own folder

    # 모델 경로 지정
    model_dir = 'model/'
    os.makedirs(model_dir, exist_ok=True)

    # 학습
    train(train_dir, validation_dir, model_dir)

    # 메모리 해제
    k.clear_session()