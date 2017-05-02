import cv2
import utils


NUMBER_OF_DIGITS = 10
roi_height = 800

# image = cv2.imread('data/Uc8a1Fm6Ksw.jpg')
image = cv2.imread('data/L6xaW5HGGbU.jpg')
# image = cv2.imread('data/v5oUefJ5QVg.jpg')
# image = cv2.imread('data/5Znd9r1_H28.jpg')

# Отображаем ROI для захвата целевого региона
digits_region = utils.get_region(image)
digits = utils.get_objects(digits_region, display_results=True, display_intermediate=False)

# utils.add_to_dataset((utils.get_unified_binary_image(digit) for digit in digits), 10, 'data/w_dataset.csv')
