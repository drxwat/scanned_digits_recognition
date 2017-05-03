import cv2
import numpy as np
import random


def get_region(image, roi_height=800):
    """
    Отображает окно для выбора интересующего региона.
    @:argument image numpy.ndarray
    @:argument roi_height integer - высота окна для выбора региона
    """

    # ХАК: Масштабируем изображение для отображения ROI не более размера экрана.
    # ПРИЧИНА: namedWindow, resizeWindow и другие методы манипуляции размером окна не работают с selectROI
    resize_scale = image.shape[0] / roi_height
    roi_width = int(image.shape[1] / resize_scale)

    roi = cv2.selectROI(cv2.resize(image, (roi_width, roi_height)), fromCenter=False)

    # Масштабируем рамку ROI для получения куска оригинального изображения
    roi_original = list(map(lambda roi_el: roi_el * resize_scale, roi))

    digits_region = image[
                    int(roi_original[1]):int(roi_original[1] + roi_original[3]),
                    int(roi_original[0]):int(roi_original[0] + roi_original[2])]

    return digits_region


def get_objects(image, display_results=False, display_intermediate=False):
    """
    Возвращает найденные с помощью watershed объекты произвольной величены (как есть)
    @:parameter image numpy.ndarray - исходное изображение
    @:parameter display_results bool  - выведет найденные объекты раскрашенные произвольным цветом
    @:return generator of numpy.ndarray - генератор обхъектов различной формы
    """

    _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Ищем background в котором уверены
    kernel = np.ones((3, 3), dtype=np.uint8)
    sure_bg = cv2.dilate(thresh, kernel, iterations=2)

    # Ищем foreground в котором уверены
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1, cv2.NORM_MINMAX)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Вычисляем неизвестную область
    unknown = cv2.subtract(sure_bg, sure_fg)

    if display_intermediate:
        cv2.imshow('Sure bg', sure_bg)
        cv2.waitKey(0)
        cv2.imshow('Sure fg', sure_fg)
        cv2.waitKey(0)
        cv2.imshow('Unknown', unknown)
        cv2.waitKey(0)

    # Инициализируем маркеры
    _, markers = cv2.connectedComponents(sure_fg)
    markers = (markers + 1)
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    colored_image = image.copy()
    colors_r = range(0, 255, 1)

    objects = []

    # Формируем координаты найденых объектов
    for i in range(2, markers.max() + 1):
        # Красим в произольные цвета найденные объекты
        colored_image[markers == i] = [random.choice(colors_r), random.choice(colors_r), random.choice(colors_r)]

        # Находим крайние пиксели объекта
        height_pixels = []
        width_pixels = []

        for row_index in range(markers.shape[0]):
            if i in markers[row_index, :]:
                height_pixels.append(row_index)

        for col_index in range(markers.shape[1]):
            if i in markers[:, col_index]:
                width_pixels.append(col_index)

        x1, y1 = width_pixels[0], height_pixels[0]
        x2, y2 = width_pixels[-1], height_pixels[-1]

        if (x2 - x1) > 3 and (y2 - y1) > 3:  # Убираем очень маленькие объекты
            objects.append((x1, y1, x2, y2))

    # Упорядочиваем объекты слева на право
    objects.sort()
    objects = merge_vertical_objects(objects)

    if display_results:
        cv2.imshow('Colored image', colored_image)
        cv2.waitKey(0)

        for x1, y1, x2, y2 in objects:
            rect = cv2.rectangle(colored_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imshow('Borders', rect)

        cv2.waitKey(0)

    return (image[y1:y2, x1:x2] for x1, y1, x2, y2 in objects)


def merge_vertical_objects(objects_coordinates, concat_percent=0.5):
    """
    Склеивает объекты по вертикали, если такие есть
    @:parameter objects_coordinates list список кортежей координат (x1, y1, x2, y2)
    @:parameter float - степень увеличения объекта для проверки включения дочерних объектов по оси x
    """

    concat_candidates = []

    # Определяем объекты, который необходимо объединить по вертикали по следующей логике
    # Если увеличеный в ширину на N% первичный объект захватывает по оси x вторичный объект, то они образуют один объект
    for i_prim, obj_primary in enumerate(objects_coordinates):
        for i_sec, obj_secondary in enumerate(objects_coordinates):

            if i_prim == i_sec:
                continue

            primary_width = obj_primary[2] - obj_primary[0]
            primary_side_addition = int((primary_width * concat_percent) / 2)
            x1, x2 = obj_primary[0] - primary_side_addition, obj_primary[2] + primary_side_addition

            # Если объект охватывает другой и нет повтора, то добавляем к кандидатам на склеивание
            if obj_secondary[0] > x1 and obj_secondary[2] < x2 and (i_sec, i_prim) not in concat_candidates:
                concat_candidates.append((i_prim, i_sec))

    def concat_objects(first, second):
        """Вычисляет координаты нового объекта скаладывая два переданых"""
        new_x1 = min(first[0], second[0])
        new_y1 = min(first[1], second[1])
        new_x2 = max(first[2], second[2])
        new_y2 = max(first[3], second[3])

        return new_x1, new_y1, new_x2, new_y2

    moved_objects = {}  # кто куда переехал
    fixed_objects = {idx: obj for idx, obj in enumerate(objects_coordinates)}

    # Объединяем помеченные объекты
    for i, (i_prim, i_sec) in enumerate(concat_candidates):
        # Объединяем объекты
        concatenated_obj = concat_objects(objects_coordinates[i_prim], objects_coordinates[i_sec])

        if i_prim in moved_objects:  # Если родительский объект уже к кому-то присоединен
            parent_id = moved_objects[i_prim]
            fixed_objects[parent_id] = concat_objects(objects_coordinates[parent_id], concatenated_obj)
            # Отмечаем куда прикрепился вторичный объект
            moved_objects[i_sec] = parent_id
        else:
            fixed_objects[i_prim] = concatenated_obj
            moved_objects[i_sec] = i_prim

        del fixed_objects[i_sec]

    return list(fixed_objects.values())


def get_unified_binary_image(image, new_shape=(30, 30)):
    """
    Унифицирует изображения по размеру. Возвращает бинарное изображение заданного размера.
    @:parameter image - numpy.ndarray
    @:parameter new_shape tuple - пара new_height и new_width
    """

    _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    unified_image = np.full(new_shape, 255, dtype=np.uint8)

    if thresh.shape[0] > new_shape[0]:
        resize_scale = thresh.shape[0] / new_shape[0]
        new_width = int(thresh.shape[1] / resize_scale)

        thresh = cv2.resize(thresh, (new_width, new_shape[0]))

    # Something gone wrong
    if thresh.shape[1] > new_shape[1]:
        raise DetectorError("This image doesn't looks like a digit. It's too wide.")

    # Размещаем изображение в центре фона
    x_offset = int(new_shape[1] / 2) - int((thresh.shape[1] / 2))
    y_offset = int(new_shape[0] / 2) - int((thresh.shape[0] / 2))

    unified_image[y_offset:y_offset + thresh.shape[0], x_offset:x_offset + thresh.shape[1]] = thresh

    return unified_image


def add_to_dataset(digits, n_digits, filename):
    """
    Автоматически маркирует и складывает в csv файл переданные объекты.
    Предполагается, что объекты отсортированы по возрастанию и переданы в кол-ве n_digits начиная с нуля.
    @:parameter digits generator of numpy.ndarray
    @:parameter n_digits integer
    @:parameter filename path to dataset file
    """

    digits = list(digits)

    if len(digits) != n_digits:
        raise DatasetGeneratorError('Passed more or less objects then expected')

    with open(filename, mode='ab') as out_file:

        for class_label in range(n_digits):

            digit = digits[class_label]
            new_shape = (1, digit.shape[0] * digit.shape[1])
            digit_vec = digit.reshape(new_shape)

            image_class = np.array([class_label], dtype=digit_vec.dtype).reshape((1, 1))
            data_sample = np.hstack((image_class, digit_vec))

            np.savetxt(out_file, data_sample, delimiter=",", fmt='%d')


# Exceptions
class UtilsError(Exception):
    def __init__(self, message):
        self.message = 'Detector error: {}'.format(message)


class DetectorError(UtilsError):
    pass


class DatasetGeneratorError(UtilsError):
    pass
