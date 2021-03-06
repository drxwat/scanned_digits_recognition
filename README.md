# Детектор и классификатор отсканированных цифр

Доступна [демонстрация](DEMO.md).

## Установка

Код был написан под python 3.5.2

### Установить pip requirements  
 
 ```
 pip install -r requirements.txt
 ```
 opencv не поставится через pip так, что можно не переживать по этому поводу.
 
### Установить opencv 3.2.0
 
 Лучше собрать из исходников. 
 
## Использование

Имеется 2 консольных утилиты. Одна рабочая, для сегментации цифр на выделенном участке изображения и их распознования. Другая вспомогательная, для автоматической маркировки сегментированных цифр и сохранения их в файл для дальнейшего обучения классификатора. 

__digits_recognizer.py__ - рабочая утилита. Пример использования можно увидеть в [демонстрации](DEMO.md).
Первым параметром передается путь к изображению, вторым путь к keras классификатору, третим путь к [csv файлу](demo/db.csv) с кодами для распознаннхы цифр.

```
python digits_recognizer.py data/L6xaW5HGGbU.jpg classifier/classifier_cnn.h5 demo/db.csv -v -es 0.4 
```

Утилита имеет ряд парметров:
1. -rh - Устанавливает высоту окна с изображением если вдруг окно не помещается на экране.
2. -dk - Size of dilate kernel. Чем больше ядро тем больше будет sure background. См. алгоритм watershed .
3. -es - Erode scale. Чем больше тем меньше будет sure foreground. См. алгоритм watershed.
4. -v и -vv - Своего рода дебаг мод. Чем больше v передано тем больше выхлопа будет.
Подробнее см. help.

__dataset_writer.py__ - вспомогательная утилита. Данная утилита __предполагает__, что ей передана область с цифрами __строго в последовательности от 0 до 9__. После сегментации сохраняет результат в файл путь к которому передан вторым парметром.

```
python dataset_writer.py data/scan1-1.jpg classifier/dataset2.csv
```
Утилита имеет такие же параметры как и digits_recognizer. Подробнее см. help.