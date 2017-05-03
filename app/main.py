import cv2
import numpy as np
from keras.models import load_model

from app import utils

ROI_HEIGHT = 800


def main():
    image = cv2.imread('digits.jpg')

    # Отображаем ROI для захвата целевого региона
    digits_region = utils.get_region(image, roi_height=ROI_HEIGHT)
    # Распиливаем на отдельные циферки
    digits = utils.get_objects(digits_region, display_results=True, display_intermediate=False)
    # Унифицируем изображения, что бы классификатор мог их понимать
    flat_digits = np.array(list(map(lambda d: utils.get_unified_binary_image(d, (30, 30)).reshape(900), digits)))

    # Классифицируем
    classifier = load_model('classifier/classifier_cnn.h5')
    classes = classifier.predict_classes(flat_digits / 255)

    print(classes)
