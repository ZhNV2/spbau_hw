Был взят ортогональный массив по ссылке http://testcover.com/pub/background/cover8.htm#3.5.6lb (Covering Array Levels for l=3, m=5, r=125, n=6), он находится в файле table1.

Далее файл 1-1.py приводит его к нашей задаче, игнорирует два последних столбца, меняет несуществующие значения на любые допустимые, удаляет одинаковые строки. Получаем таблицу тестов table2.

Далее файл 1-2.py проверяет каждый тест на возможность удаления. Для этого мы считаем сколько раз встречается каждая тройка параметров во всех тестах. Далее если для теста хотя бы одна из его четырех троек встречается всего 1 раз в наших тестах, то его выкинуть нельзя. Итоговые тесты и комментарии с возможностью удаления находятся в table3.
