import cv2
import numpy as np
from keras.models import load_model
import argparse
import utils
import csv

# Параметры по умолчанию
roi_height = 800
dilate_kernel_size = 3
erode_scale = 0.05

ap = argparse.ArgumentParser(description='Detects and recognizes digits in selected area of passed file.')
ap.add_argument('image', help='Path to the image')
ap.add_argument('classifier', help='Path to keras classifier model. Typically it *.h5 file.')
ap.add_argument('db', help='Path to DB csv file.')
ap.add_argument('-rh', type=int, default=roi_height,
                help='ROI window height (default: {})'.format(roi_height))
ap.add_argument('-dk', type=int, default=dilate_kernel_size,
                help='Size of dilate kernel in order to get sure background for watershed algorithm of segmentation. '
                     'Typically from 2 to 4 (default {}).'.format(dilate_kernel_size))
ap.add_argument('-es', type=float, default=erode_scale,
                help='Erode scale in order to get sure foreground for watershed algorithm of segmentation.'
                     'Typically from 0.05 to 0.4 (default {})'.format(erode_scale))
ap.add_argument('-v', action='store_true', help='Verbose')
ap.add_argument('-vv', action='store_true', help='Even more verbose')


args = ap.parse_args()
path_to_image, path_to_classifier, path_to_db, roi_height, dilate_kernel_size, erode_scale, v, vv = \
    args.image, args.classifier, args.db, args.rh, args.dk, args.es, args.v, args.vv
image = cv2.imread(path_to_image)

# Отображаем ROI для захвата целевого региона
digits_region = utils.get_region(image, roi_height=roi_height)
# Распиливаем на отдельные циферки
print('Segmenting...')

digits = utils.get_objects(digits_region, dilate_kernel_size=dilate_kernel_size, erode_scale=erode_scale,
                           display_results=True if v or vv else False, display_intermediate=True if vv else False)

# Унифицируем изображения, что бы классификатор мог их понимать
unified_digits = np.array(list(map(lambda d: utils.get_unified_binary_image(d, (30, 30)), digits)))
unified_digits = unified_digits.reshape((unified_digits.shape[0], 30, 30, 1))

print('Found {} objects.'.format(len(unified_digits)))

if len(unified_digits) > 0:
    # Классифицируем
    print('Classifying...')

    classifier = load_model(path_to_classifier)
    detected_digits = [str(d) for d in classifier.predict_classes(unified_digits / 255)]
    digits_string = ''.join(detected_digits)

    print('Detected digits: {}'.format(digits_string))
    print('Searching for code in DB...')

    codes_n = 0
    with open(path_to_db) as db_file:
        db_reader = csv.reader(db_file)
        for i, row in enumerate(db_reader):
            if row[0] == digits_string:
                codes_n += 1
                print('For digits {} found code {} on line {}'.format(digits_string, row[1], i+1))
        if codes_n == 0:
            print('No one code found for detected digits: {}'.format(digits_string))
else:
    print('No one digit found in the selected area. We are so sorry.')
