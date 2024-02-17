import os
import matplotlib.pyplot as plt
import numpy as np
import warnings

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from glob import glob
from tkinter import messagebox


def mac_learn_step():
    warnings.filterwarnings("ignore")

    plt.show()

    image_size = [224, 224]
    valid_path = "/Users/burakkeles/PycharmProjects/car_brand_predictor/Images/Test/"
    train_path = "/Users/burakkeles/PycharmProjects/car_brand_predictor/Images/Train/"

    resnet = ResNet50(include_top=False,
                      input_shape=image_size + [3],
                      weights='imagenet')
    plot_model(resnet)

    for layer in resnet.layers:
        layer.trainable = False

    folders = glob("/Users/burakkeles/PycharmProjects/car_brand_predictor/Images/Test/*")
    print(folders)

    x = Flatten()(resnet.output)
    prediction = Dense(len(folders), activation='softmax')(x)

    model = Model(inputs=resnet.input,
                  outputs=prediction)
    model.summary()
    plot_model(model)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True, )
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(train_path,
                                                     target_size=[224, 224],
                                                     batch_size=32,
                                                     class_mode='categorical')
    test_set = test_datagen.flow_from_directory(valid_path,
                                                target_size=[224, 224],
                                                batch_size=32,
                                                class_mode='categorical')

    r = model.fit_generator(training_set,
                            validation_data=test_set,
                            epochs=50,
                            steps_per_epoch=len(training_set),
                            validation_steps=len(test_set))

    plt.plot(r.history['loss'], label='train_loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    # plot the accuracy

    plt.plot(r.history['accuracy'], label='train_acc ')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')

    # checking directory path and creation

    work_dir = "/Users/burakkeles/PycharmProjects/car_brand_predictor/working"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    model.save("/Users/burakkeles/PycharmProjects/car_brand_predictor/working/model_resnet50.h5")
    y_pred = model.predict(test_set)
    print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)


def load_model_and_predict(image_source, model_source):
    model = load_model(model_source)

    img = image.load_img(image_source, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255

    preds = model.predict(x)
    print("Raw predictions:", preds)
    preds = np.argmax(preds, axis=1)
    print(preds)

    if preds == 0:
        preds = "The car is Audi"
    elif preds == 1:
        preds = "The car is Lamborgini"
    else:
        preds = "The car is Mercedes"

    messagebox.showinfo("result :", preds)


def learner_control_train():
    mac_learn_step()

