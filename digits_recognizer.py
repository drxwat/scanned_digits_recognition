import cv2
import numpy as np
from keras.models import load_model
import argparse
import utils

# Параметры по умолчанию
roi_height = 800

ap = argparse.ArgumentParser(description='Detects and recognizes digits in selected area of passed file.')
ap.add_argument('image', help='Path to the image')
ap.add_argument('-rh', type=int, default=roi_height, metavar='ROI',
                help='ROI window height (default: {})'.format(roi_height))

args = ap.parse_args()
path_to_image, roi_height = args.image, args.rh

image = cv2.imread(path_to_image)

# Отображаем ROI для захвата целевого региона
digits_region = utils.get_region(image, roi_height=roi_height)
# Распиливаем на отдельные циферки
digits = utils.get_objects(digits_region, display_results=True, display_intermediate=False)
# Унифицируем изображения, что бы классификатор мог их понимать
flat_digits = np.array(list(map(lambda d: utils.get_unified_binary_image(d, (30, 30)).reshape(900), digits)))

# Классифицируем
classifier = load_model('classifier/classifier_cnn.h5')
classes = classifier.predict_classes(flat_digits / 255)

print(classes)