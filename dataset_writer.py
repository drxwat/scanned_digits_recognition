import cv2
import argparse
import utils

# Параметры по умолчанию
roi_height = 800
dilate_kernel_size = 3
erode_scale = 0.05

ap = argparse.ArgumentParser(description='Detects and writes digits from 0 to 9 to output file. '
                                         'Assumes that selected digits printed in exact order from 0 to 9')
ap.add_argument('image', help='Path to the image')
ap.add_argument('output', help='Path to output csv file')
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
path_to_image, path_to_output, roi_height, dilate_kernel_size, erode_scale, v, vv = \
    args.image, args.output, args.rh, args.dk, args.es, args.v, args.vv

image = cv2.imread(path_to_image)

# Отображаем ROI для захвата целевого региона
digits_region = utils.get_region(image, roi_height=roi_height)

try:
    # Распиливаем на отдельные циферки
    digits = utils.get_objects(digits_region, dilate_kernel_size=dilate_kernel_size, erode_scale=erode_scale,
                               display_results=True if v or vv else False, display_intermediate=True if vv else False)
    # Маркирует цифры от 0 до 9 и складывает их в файл
    utils.add_to_dataset((utils.get_unified_binary_image(digit) for digit in digits),
                         n_digits=10,
                         filename=path_to_output)
except utils.DetectorError as err:
    print(err.message)
    exit(1)
except utils.DatasetGeneratorError as err:
    print(err.message)
    exit(1)
