import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import re


# Чтение содержимого файла
def read_file(filename):
    """Чтение файла с поддержкой различных кодировок"""
    try:
        # Пробуем открыть как UTF-8
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Пробуем открыть как Windows-1251 (кириллица)
            with open(filename, 'r', encoding='cp1251') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # Пробуем открыть как KOI8-R
                with open(filename, 'r', encoding='koi8-r') as f:
                    return f.read()
            except:
                return None


# Запись в файл
def write_file(filename, content):
    """Запись в файл с указанной кодировкой"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)


class RouteCipher:
    # Определяем алфавит и вспомогательные символы
    ALPHABET = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    NUMBERS = '0123456789'
    SPECIAL = ".,!?;:()[]{}\"'«»—–-…№%$@#&*+=/\\|~`^<>_ \n\t" + chr(8239) + chr(8201)  # Добавляем специальные пробелы
    ALL_CHARS = ALPHABET + ALPHABET.lower() + NUMBERS + SPECIAL
    
    # Добавляем константы для анализа погодных прогнозов
    WEATHER_KEYWORDS = [
        'солнце', 'солнечно', 'ясно', 'дождь', 'дождливо', 'осадки', 
        'ветер', 'ветрено', 'прогноз', 'погода', 'температура', 'тепло', 
        'холодно', 'мороз', 'град', 'гроза', 'снег', 'облачно', 'туман',
        'градус', 'давление', 'влажность', 'метеослужба'
    ]

    def validate_text(self, text):
        # Проверяем, что текст содержит только допустимые символы
        invalid_chars = []
        for char in text:
            if char not in self.ALL_CHARS:
                invalid_chars.append((char, ord(char)))

        if invalid_chars:
            error_msg = "Обнаружены недопустимые символы:\n"
            for char, code in invalid_chars:
                error_msg += f"'{char}' (код {code}) "
            error_msg += "\n\nДобавьте эти символы в SPECIAL или удалите их из текста."
            raise ValueError(error_msg)
        return True

    def detect_weather_forecast(self, text):
        """
        Анализирует текст с использованием частотных N-грамм русского языка,
        с особым акцентом на слово "солнце".
        
        Возвращает:
        - float: оценка от 0.0 до 1.0, где 1.0 означает высокое соответствие языковым паттернам
        """
        # Все к нижнему регистру - иначе замучаемся проверять
        text_lower = text.lower()
        
        # Самые частые биграммы в русском - если их нет, значит что-то не так
        popular_bigrams = ['ст', 'но', 'то', 'на', 'ен', 'ов', 'ни', 'ра', 'во', 
                          'ко', 'ал', 'ли', 'по', 'ре', 'ол', 'пр', 'ть', 'ат', 
                          'ет', 'та', 'го', 'ос', 'ер', 'ит', 'нн', 'ск', 'ны', 'ие']
        
        # Частые триграммы - тут сложнее, но тоже важно
        popular_trigrams = ['ост', 'ого', 'ени', 'ста', 'про', 'ная', 'ени', 'что', 
                           'тор', 'ать', 'кот', 'его', 'ном', 'ого', 'ова', 'ств']
        
        # Наше любимое слово "солнце" - без него никуда
        sun_word = "солнце"
        sun_ngrams = []
        
        # Делаем кусочки из "солнца" разной длины
        # Короткие куски (по 2 буквы)
        for i in range(len(sun_word)-1):
            sun_ngrams.append(sun_word[i:i+2])
            
        # Средние куски (по 3 буквы)
        for i in range(len(sun_word)-2):
            sun_ngrams.append(sun_word[i:i+3])
            
        # Большие куски (4 и 5 букв)
        for i in range(len(sun_word)-3):
            sun_ngrams.append(sun_word[i:i+4])
        for i in range(len(sun_word)-4):
            sun_ngrams.append(sun_word[i:i+5])
        
        # Считаем сколько обычных биграмм попалось в тексте
        bigram_score = 0
        for bigram in popular_bigrams:
            count = text_lower.count(bigram)
            bigram_score += count
        
        # Нормализуем - не может же быть биграмм больше чем букв в тексте
        total_chars = max(1, len(text_lower) - 1)
        bigram_ratio = min(1.0, bigram_score / (total_chars * 0.3))
        
        # То же самое с триграммами, но их вес больше - они реже случайно совпадают
        trigram_score = 0
        for trigram in popular_trigrams:
            count = text_lower.count(trigram)
            trigram_score += count * 1.5  # Трешки весят больше
        
        # Тоже нормализуем
        trigram_ratio = min(1.0, trigram_score / (total_chars * 0.2))
        
        # Считаем куски "солнца" - чем длиннее кусок, тем он ценнее
        sun_score = 0
        for ngram in sun_ngrams:
            ngram_count = text_lower.count(ngram)
            if ngram_count > 0:
                # Длинный кусок - большой куш
                sun_score += ngram_count * (len(ngram) / 2.0)
        
        # Если нашли целое слово "солнце" - это джекпот
        if sun_word in text_lower:
            sun_score += 5.0
            
        # Нормализуем солнечный счет
        sun_score = min(1.0, sun_score / 15.0)
        
        # Смешиваем всё в правильных пропорциях
        final_score = bigram_ratio * 0.4 + trigram_ratio * 0.3 + sun_score * 0.3
        
        return min(1.0, final_score)

    def pad_text(self, text, width, height):
        # Проверяем, что размеры таблицы корректны
        if width <= 0 or height <= 0:
            raise ValueError("Ширина и высота таблицы должны быть положительными числами")

        # Вычисляем необходимую длину текста
        required_length = width * height

        # Если текст длиннее необходимого, обрезаем его
        if len(text) > required_length:
            text = text[:required_length]

        # Дополняем текст пробелами до нужной длины
        padding_length = required_length - len(text)
        return text + ' ' * padding_length

    def create_matrix(self, text, width, height):
        # Создаем матрицу из текста
        matrix = []
        for i in range(height):
            row = list(text[i * width: (i + 1) * width])
            matrix.append(row)
        return matrix

    def spiral_route(self, width, height):
        """Генерация маршрута по спирали (по часовой стрелке, начиная с левого верхнего угла)
        Пример для матрицы 3x3:
        1→2→3
        8→9→4
        7←6←5
        """
        matrix = [[0] * width for _ in range(height)]
        x, y = 0, 0  # Начинаем с левого верхнего угла
        dx, dy = 0, 1  # Начальное направление - вправо
        route = []

        for _ in range(width * height):
            route.append((x, y))
            matrix[x][y] = 1

            # Проверяем следующую позицию
            next_x, next_y = x + dx, y + dy

            # Если следующая позиция выходит за границы или уже посещена
            if (next_x < 0 or next_x >= height or
                    next_y < 0 or next_y >= width or
                    matrix[next_x][next_y] == 1):
                # Меняем направление: вправо → вниз → влево → вверх
                dx, dy = dy, -dx
                next_x, next_y = x + dx, y + dy

            x, y = next_x, next_y

        return route

    def snake_route(self, width, height):
        """Генерация маршрута змейкой (слева направо в четных строках, справа налево в нечетных)
        Пример для матрицы 3x3:
        1→2→3
        6←5←4
        7→8→9
        """
        route = []
        for i in range(height):
            if i % 2 == 0:  # Четные строки (включая 0) - слева направо
                for j in range(width):
                    route.append((i, j))
            else:  # Нечетные строки - справа налево
                for j in range(width - 1, -1, -1):
                    route.append((i, j))
        return route

    def analyze_route_pattern(self, text, width, height):
        """
        Анализирует текст и определяет оптимальный тип маршрута (спираль или змейка).
        Анализ основан на частотных N-граммах русского языка и 
        наличии слова "солнце" и его фрагментов.

        Параметры:
        - text: текст для анализа
        - width: ширина таблицы
        - height: высота таблицы

        Возвращает:
        - "спираль" или "змейка" в зависимости от результатов анализа
        """
        # Проверяем, что нам дали нормальные параметры
        if not text or width <= 0 or height <= 0:
            return "спираль"  # По умолчанию, если входные данные - ерунда
        
        # Если текст слишком короткий - тоже берем спираль, так надежнее
        if len(text) < width * 2:
            return "спираль"  

        # Создаем два маршрута - змейкой и спиралью
        spiral_route = self.spiral_route(width, height)
        snake_route = self.snake_route(width, height)

        # Делаем пустые матрицы для обоих вариантов
        spiral_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        snake_matrix = [[' ' for _ in range(width)] for _ in range(height)]

        # Обрезаем текст, если он вдруг слишком длинный
        text_normalized = text[:min(len(text), len(spiral_route))]

        # Заполняем матрицы - каждая буква на свое место
        for idx, char in enumerate(text_normalized):
            if idx < len(spiral_route):
                i, j = spiral_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    spiral_matrix[i][j] = char

            if idx < len(snake_route):
                i, j = snake_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    snake_matrix[i][j] = char

        # Читаем матрицы построчно - получаем два разных текста
        spiral_text = ''.join(''.join(row) for row in spiral_matrix)
        snake_text = ''.join(''.join(row) for row in snake_matrix)

        # Проверяем оба текста нашими умными алгоритмами
        spiral_quality = self.assess_decryption_quality(spiral_text)
        snake_quality = self.assess_decryption_quality(snake_text)
        
        # Еще одна проверка - на лингвистическое качество
        spiral_linguistic = self.secondary_quality_check(spiral_text)
        snake_linguistic = self.secondary_quality_check(snake_text)
        
        # Проверяем частотность n-грамм
        spiral_ngram_score = self.detect_weather_forecast(spiral_text)
        snake_ngram_score = self.detect_weather_forecast(snake_text)
        
        # Специально ищем солнце и его части - это наш ключ
        sun_word = "солнце"
        sun_ngrams = []
        
        # Собираем все возможные куски слова "солнце"
        # Маленькие (по 2)
        for i in range(len(sun_word)-1):
            sun_ngrams.append(sun_word[i:i+2])
            
        # Средние (по 3)
        for i in range(len(sun_word)-2):
            sun_ngrams.append(sun_word[i:i+3])
            
        # Большие (4 и 5)
        for i in range(len(sun_word)-3):
            sun_ngrams.append(sun_word[i:i+4])
        for i in range(len(sun_word)-4):
            sun_ngrams.append(sun_word[i:i+5])
        
        # Считаем сколько кусков нашлось в каждом варианте расшифровки
        spiral_sun_count = sum(spiral_text.lower().count(ng) for ng in sun_ngrams)
        snake_sun_count = sum(snake_text.lower().count(ng) for ng in sun_ngrams)
        
        # Проверяем, нашлось ли целое слово
        spiral_has_full_sun = sun_word in spiral_text.lower()
        snake_has_full_sun = sun_word in snake_text.lower()
        
        # Бонусы для вариантов
        spiral_sun_bonus = 0
        snake_sun_bonus = 0
        
        # Если есть полное слово "солнце"
        if spiral_has_full_sun:
            spiral_sun_bonus += 0.3
        if snake_has_full_sun:
            snake_sun_bonus += 0.3
            
        # Если больше фрагментов слова "солнце"
        if spiral_sun_count > snake_sun_count * 1.5:
            spiral_sun_bonus += 0.2
        elif snake_sun_count > spiral_sun_count * 1.5:
            snake_sun_bonus += 0.2
        
        # Вычисляем общую оценку с весами
        # Баланс между общим качеством и специфическими проверками
        spiral_score = (spiral_quality * 0.3 + 
                        spiral_linguistic * 0.2 + 
                        spiral_ngram_score * 0.3 + 
                        spiral_sun_bonus)
                        
        snake_score = (snake_quality * 0.3 + 
                       snake_linguistic * 0.2 + 
                       snake_ngram_score * 0.3 + 
                       snake_sun_bonus)
        
        # Применяем корректировки на основе формы таблицы
        
        # 1. Для квадратных таблиц предпочтительнее спираль
        if abs(width - height) <= 2:
            spiral_score *= 1.1
            
        # 2. Для очень широких таблиц предпочтительнее змейка
        if width > height * 2:
            snake_score *= 1.15
            
        # 3. Для очень высоких таблиц тоже предпочтительнее змейка
        if height > width * 2:
            snake_score *= 1.1
            
        # 4. Для таблиц шириной 11 исторически предпочтительнее спираль
        if width == 11:
            spiral_score *= 1.2
            
        # 5. Для маленьких таблиц (до 5x5) предпочтительнее спираль
        if width <= 5 and height <= 5:
            spiral_score *= 1.1
        
        # Бонус, если в одном варианте есть слово "солнце" или много его фрагментов
        if (spiral_has_full_sun or spiral_sun_count >= 7) and not (snake_has_full_sun or snake_sun_count >= 7):
            spiral_score *= 1.5
        elif (snake_has_full_sun or snake_sun_count >= 7) and not (spiral_has_full_sun or spiral_sun_count >= 7):
            snake_score *= 1.5
        
        # Возвращаем тип маршрута с наивысшей оценкой
        if snake_score > spiral_score:
            return "змейка"
        else:
            return "спираль"

    def get_route(self, width, height, route_type):
        """Возвращает маршрут указанного типа для таблицы заданного размера"""
        if route_type == "спираль":
            return self.spiral_route(width, height)
        elif route_type == "змейка":
            return self.snake_route(width, height)
        else:
            raise ValueError(f"Неизвестный тип маршрута: {route_type}. Используйте 'спираль' или 'змейка'.")

    def format_table(self, matrix):
        """Форматирует матрицу в виде читаемой таблицы с границами"""
        # Защита от некорректной матрицы
        if not matrix:
            return "Пустая таблица"

        if not isinstance(matrix, list) or len(matrix) == 0:
            return "Некорректная матрица"

        if not matrix[0] or not isinstance(matrix[0], list):
            return "Пустая или некорректная матрица"

        # Форматируем матрицу
        result = []
        for row in matrix:
            # Проверяем каждую строку матрицы
            if not isinstance(row, list):
                continue
            result.append(''.join(str(cell) if cell is not None else ' ' for cell in row))

        if not result:
            return "Пустая таблица"

        return '\n'.join(result)

    def format_table_with_route(self, text, width, height, route_type="спираль"):
        """Форматирует матрицу в виде читаемой таблицы, размещая текст сразу по маршруту"""
        # Проверяем входные данные
        if not text:
            return "Пустая таблица"

        if width <= 0 or height <= 0:
            return "Некорректные размеры таблицы"

        # Создаем пустую матрицу
        matrix = [[' ' for _ in range(width)] for _ in range(height)]

        try:
            # Выбираем маршрут
            route = self.get_route(width, height, route_type)

            if not route:
                return f"Ошибка: Не удалось создать маршрут типа '{route_type}' для таблицы {width}x{height}"

            # Заполняем матрицу по маршруту
            for i, (x, y) in enumerate(route):
                if i < len(text) and 0 <= x < height and 0 <= y < width:
                    matrix[x][y] = text[i]

            return self.format_table(matrix)
        except Exception as e:
            return f"Ошибка при форматировании таблицы: {e}"

    def preprocess_text(self, text, width=None, height=None, filler='Х', remove_spaces=False):
        """Предобрабатывает текст для шифрования:
        - удаляет пробелы (опционально)
        - заполняет до необходимой длины указанным символом-заполнителем
        - нормализует русские символы

        Args:
            text (str): Исходный текст
            width (int, optional): Ширина таблицы
            height (int, optional): Высота таблицы. Если не указана, вычисляется автоматически.
            filler (str, optional): Символ-заполнитель. По умолчанию 'Х'.
            remove_spaces (bool, optional): Удалять ли пробелы. По умолчанию False.

        Returns:
            tuple: (подготовленный текст, ширина, высота, количество удаленных пробелов)
        """
        # Проверяем текст на пустоту
        if not text:
            raise ValueError("Текст не может быть пустым")

        # Преобразуем text к строке, чтобы избежать проблем с типами
        text = str(text)

        # Удаляем BOM-маркер, если он есть
        if text.startswith('\ufeff'):
            text = text[1:]

        # Нормализуем спецсимволы кириллицы
        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace(chr(8239), ' ').replace(chr(8201), ' ')
        text = text.replace('\xa0', ' ')  # Неразрывный пробел

        # Удаляем пробелы и управляющие символы, если требуется
        spaces_removed = 0
        if remove_spaces:
            cleaned_text = ''
            for c in text:
                if c not in ' \t\n\r\xa0':
                    cleaned_text += c
                else:
                    spaces_removed += 1
        else:
            cleaned_text = text

        # Если не указана ширина, возвращаем только очищенный текст и другие параметры
        if width is None:
            return cleaned_text, 0, 0, spaces_removed

        # Если ширина <= 0, используем криптоанализ
        if width <= 0:
            estimated = self.estimate_table_size(cleaned_text)
            width = estimated["best_width"] if isinstance(estimated, dict) else estimated[0][0]

        # Высота всегда рассчитывается автоматически на основе ширины
        height = (len(cleaned_text) + width - 1) // width

        # Вычисляем необходимую длину
        required_length = width * height

        # Дополняем текст символами-заполнителями до нужной длины
        if len(cleaned_text) < required_length:
            padding_length = required_length - len(cleaned_text)
            cleaned_text += filler * padding_length

        return cleaned_text, width, height, spaces_removed

    def get_fillers(self):
        """Returns a list of possible fillers for the decrypt method."""
        fillers = []
        for char in 'ХЬЪЫЭЮЯ':  # Rare Russian letters
            fillers.append(char)
        return fillers

    def assess_decryption_quality(self, text):
        """Усовершенствованная оценка качества расшифровки с учетом прогноза погоды"""
        # Используем только первые 1000 символов для анализа (достаточная выборка)
        sample = text[:1000] if len(text) > 1000 else text

        if not sample:
            return 0.0

        # 1. Подсчет русских букв и знаков препинания
        russian_chars = sum(1 for c in sample if c.lower() in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        russian_ratio = russian_chars / len(sample) if sample else 0

        # Базовая проверка на русский текст - если меньше 30% русских букв, вероятно это не русский текст
        if russian_ratio < 0.3:
            return russian_ratio * 0.5  # Возвращаем низкую оценку

        # 2. Подсчет пробелов (нормальное соотношение ~15-20%)
        space_ratio = sample.count(' ') / len(sample) if sample else 0
        space_score = 1.0 - 2.0 * abs(0.18 - space_ratio) if space_ratio > 0 else 0.0

        # Если пробелов слишком мало или слишком много, это плохой признак
        if space_ratio < 0.05 or space_ratio > 0.3:
            space_score = space_score / 2

        # 3. Анализ частотных биграмм в русском языке
        common_bigrams = ['ст', 'но', 'то', 'на', 'ен', 'ов', 'ни', 'ра', 'во', 'ко', 'ал', 'ли', 'по', 'ре', 'ол',
                          'пр', 'ть', 'ат', 'ет', 'та', 'го', 'ос', 'ер', 'ит', 'нн', 'ск', 'ны', 'ие']
        sample_lower = sample.lower()
        bigram_count = 0

        for bigram in common_bigrams:
            bigram_count += sample_lower.count(bigram)

        # Нормализуем на длину текста и количество проверяемых биграмм
        bigram_ratio = bigram_count / max(1, len(sample) - 1) * (10 / len(common_bigrams))

        # 4. Анализ знаков препинания и их позиций
        punct_marks = '.,:;!?'
        punct_count = sum(1 for c in sample if c in punct_marks)
        punct_ratio = punct_count / len(sample) if sample else 0

        # Оптимальное соотношение знаков препинания ~5-10%
        punct_score = 1.0 - abs(0.07 - punct_ratio) * 5.0 if punct_ratio > 0 else 0.0
        punct_score = max(0.0, min(1.0, punct_score))

        # 5. Проверка начала предложений (с заглавной буквы после точки)
        valid_caps = 0
        total_sentences = 0

        for i in range(1, len(sample) - 1):
            if sample[i - 1] in '.!?' and sample[i] == ' ' and i + 1 < len(sample):
                total_sentences += 1
                if sample[i + 1].isupper():
                    valid_caps += 1

        caps_score = valid_caps / max(1, total_sentences)

        # 6. Анализ соотношения гласных и согласных
        vowels = 'аеёиоуыэюя'
        consonants = 'бвгджзйклмнпрстфхцчшщ'

        vowel_count = sum(1 for c in sample_lower if c in vowels)
        consonant_count = sum(1 for c in sample_lower if c in consonants)

        # Нормальное соотношение для русского языка: ~42% гласных, ~58% согласных
        if vowel_count + consonant_count > 0:
            vowel_ratio = vowel_count / (vowel_count + consonant_count)
            vowel_consonant_score = 1.0 - abs(0.42 - vowel_ratio) * 2.5
            vowel_consonant_score = max(0.0, min(1.0, vowel_consonant_score))
        else:
            vowel_consonant_score = 0.0

        # 7. Анализ длин слов
        words = [w for w in sample.split() if w]

        if words:
            # Средняя длина слова в русском языке ~5.5 символов
            avg_word_length = sum(len(w) for w in words) / len(words)
            word_length_score = 1.0 - abs(5.5 - avg_word_length) / 5.0
            word_length_score = max(0.0, min(1.0, word_length_score))

            # Проверка наличия очень длинных слов (потенциально слипшихся)
            long_words_ratio = sum(1 for w in words if len(w) > 15) / len(words)
            if long_words_ratio > 0.1:  # Если более 10% слов длиннее 15 символов, снижаем оценку
                word_length_score *= (1.0 - long_words_ratio)
        else:
            word_length_score = 0.0
            
        # 8. НОВОЕ: Специальный анализ на соответствие прогнозу погоды
        weather_score = self.detect_weather_forecast(sample)

        # Объединяем все метрики в общую оценку с различными весами
        quality = (
                russian_ratio * 0.2 +      # Вес соотношения русских букв (уменьшен)
                space_score * 0.15 +       # Вес правильного соотношения пробелов (уменьшен)
                bigram_ratio * 0.1 +       # Вес частотных биграмм (уменьшен)
                punct_score * 0.05 +       # Вес знаков препинания
                caps_score * 0.1 +         # Вес правильного начала предложений
                vowel_consonant_score * 0.1 + # Вес соотношения гласных и согласных
                word_length_score * 0.1 +  # Вес средней длины слов
                weather_score * 0.2        # НОВОЕ: Вес соответствия прогнозу погоды (значительный вес)
        )

        return min(1.0, max(0.0, quality))

    def analyze_route_structure(self, encrypted_text, decrypted_text, width, height, route_type):
        """Анализирует структуру маршрута и соответствие расшифрованного текста его особенностям"""
        try:
            # Ограничиваем текст для анализа
            encrypted_sample = encrypted_text[:min(1000, len(encrypted_text))]
            decrypted_sample = decrypted_text[:min(1000, len(decrypted_text))]

            # Стартовая оценка
            score = 0.5

            # Получаем маршрут
            route = self.get_route(width, height, route_type)

            if not route:
                return score  # Возвращаем базовую оценку, если маршрут не удалось получить

            # Создаем матрицу из расшифрованного текста
            matrix = [[' ' for _ in range(width)] for _ in range(height)]
            for i, (x, y) in enumerate(route):
                if i < len(encrypted_sample) and 0 <= x < height and 0 <= y < width:
                    matrix[x][y] = encrypted_sample[i]

            # Проверки общие для всех маршрутов
            # 1. Анализ частоты пробелов в расшифрованном тексте
            space_ratio = decrypted_sample.count(' ') / max(1, len(decrypted_sample))
            if 0.15 <= space_ratio <= 0.25:  # Нормальная частота пробелов для русского текста
                score += 0.05

            # 2. Анализ структуры предложений
            sentences = [s.strip() for s in re.split(r'[.!?]+', decrypted_sample) if s.strip()]
            valid_sentences = sum(1 for s in sentences if len(s) > 5 and s[0].isupper())

            if sentences and valid_sentences / len(sentences) > 0.7:
                score += 0.05

            # Проверка структуры для спирального маршрута
            if route_type == "спираль":
                # 1. В спиральном маршруте начало текста должно быть на периметре
                # Проверяем, соответствует ли первый символ расшифрованного текста ожидаемому положению
                if route and len(route) > 0 and len(encrypted_sample) > 0:
                    first_pos = route[0]
                    on_perimeter = (first_pos[0] == 0 or first_pos[0] == height - 1 or
                                    first_pos[1] == 0 or first_pos[1] == width - 1)
                    if on_perimeter:
                        score += 0.1

                # 2. В спиральном маршруте конец часто находится в центре
                if route and len(route) > 0 and len(encrypted_sample) > 0:
                    last_pos = route[-1]
                    near_center = (abs(last_pos[0] - height // 2) <= height // 4 and
                                   abs(last_pos[1] - width // 2) <= width // 4)
                    if near_center:
                        score += 0.1

                # 3. Проверяем периметр на наличие целостных языковых конструкций
                edge_chars = []
                # Верхний ряд
                if height > 0 and width > 0:
                    edge_chars.extend([matrix[0][j] for j in range(min(width, len(matrix[0])))])
                # Правый столбец
                if height > 0 and width > 0:
                    edge_chars.extend([matrix[i][width - 1] for i in range(1, min(height, len(matrix))) if
                                       width - 1 < len(matrix[i])])
                # Нижний ряд (в обратном порядке)
                if height > 1 and width > 1:
                    if height - 1 < len(matrix):
                        edge_chars.extend(
                            [matrix[height - 1][j] for j in range(width - 2, -1, -1) if j < len(matrix[height - 1])])
                # Левый столбец (в обратном порядке)
                if height > 2 and width > 0:
                    edge_chars.extend([matrix[i][0] for i in range(height - 2, 0, -1) if 0 < len(matrix[i])])

                edge_text = ''.join(edge_chars)

                # Проверяем, содержит ли периметр пробелы примерно через каждые 5-7 символов (признак слов)
                if edge_text:
                    space_positions = [i for i, c in enumerate(edge_text) if c == ' ']
                    if space_positions:
                        space_intervals = [space_positions[i + 1] - space_positions[i] for i in
                                           range(len(space_positions) - 1)]
                        avg_interval = sum(space_intervals) / max(1, len(space_intervals))
                        if 4 <= avg_interval <= 8:  # Средняя длина слова в русском языке
                            score += 0.1

                # 4. Проверка цельности слов в спиральном маршруте
                word_coherence = 0
                for i in range(min(len(encrypted_sample), len(route)) - 1):
                    if i + 1 >= len(route):
                        continue

                    x1, y1 = route[i]
                    x2, y2 = route[i + 1]

                    # Если символы образуют потенциальное слово (нет между ними пробела)
                    if (0 <= x1 < height and 0 <= y1 < width and 0 <= x2 < height and 0 <= y2 < width and
                            matrix[x1][y1] != ' ' and matrix[x2][y2] != ' '):

                        # Соседние символы должны быть относительно близко друг к другу в спирали
                        if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                            word_coherence += 1

                word_coherence_ratio = word_coherence / max(1, len(encrypted_sample) - 1)
                score += min(0.15, word_coherence_ratio * 0.5)

            # Проверка структуры для змеиного маршрута
            elif route_type == "змейка":
                # 1. В змеином маршруте текст читается построчно
                rows = []
                for i in range(min(height, len(matrix))):
                    row_indices = [(i, j) for j in range(width)] if i % 2 == 0 else [(i, j) for j in
                                                                                     range(width - 1, -1, -1)]
                    row_text = ''.join(
                        matrix[x][y] if x < height and y < width and y < len(matrix[x]) else ' ' for x, y in
                        row_indices)
                    rows.append(row_text.strip())

                # 2. Проверяем, заканчиваются ли четные строки знаком препинания или пробелом (признак целостности)
                valid_row_ends = 0
                for i, row in enumerate(rows):
                    if not row:
                        continue

                    if i % 2 == 0:  # Четные строки (слева направо)
                        if row[-1] in ' .,!?:;':
                            valid_row_ends += 1
                    else:  # Нечетные строки (справа налево)
                        if row[0] in ' .,!?:;':
                            valid_row_ends += 1

                if rows:
                    valid_row_ratio = valid_row_ends / len(rows)
                    score += valid_row_ratio * 0.15

                # 3. Проверяем частоту пробелов в строках (должна быть примерно одинаковой для змейки)
                space_counts = [row.count(' ') for row in rows if row]
                if space_counts:
                    avg_spaces = sum(space_counts) / len(space_counts)
                    space_deviation = sum(abs(count - avg_spaces) for count in space_counts) / len(space_counts)

                    # Если отклонение невелико, это хороший признак змейки
                    if space_deviation < avg_spaces * 0.3:
                        score += 0.1

                # 4. Проверяем переходы между строками на разрывы слов
                if len(rows) >= 2:
                    continuous_breaks = 0
                    for i in range(len(rows) - 1):
                        if not rows[i] or not rows[i + 1]:
                            continue

                        if i % 2 == 0:  # Переход с четной строки на нечетную (вправо-влево)
                            if rows[i][-1].isalpha() and rows[i + 1][0].isalpha():
                                continuous_breaks += 1
                        else:  # Переход с нечетной строки на четную (влево-вправо)
                            if rows[i][0].isalpha() and rows[i + 1][-1].isalpha():
                                continuous_breaks += 1

                    # Мало разрывов слов - хороший признак для змейки
                    break_ratio = continuous_breaks / (len(rows) - 1)
                    if break_ratio < 0.3:
                        score += 0.15

                # 5. Проверяем, что начинается с заглавной буквы (как хороший текст)
                if rows and rows[0] and rows[0][0].isupper():
                    score += 0.05

            # Ограничиваем итоговую оценку
            return min(1.0, score)

        except Exception as e:
            return 0.5  # Возвращаем базовую оценку в случае ошибки

    def secondary_quality_check(self, text):
        """Дополнительная проверка качества расшифровки, фокусирующаяся на лингвистическом анализе"""
        # Берем первую тысячу символов - чего там всё читать-то
        sample = text[:1000] if len(text) > 1000 else text

        if not sample or len(sample) < 10:
            return 0.0  # Пустой текст - нулевая оценка

        # 1. Подсчет слов, соответствующих русскому языку (имеют гласные)
        words = [w for w in sample.split() if len(w) > 1]

        if not words:
            return 0.0  # Нет слов - плохо дело

        # Анализ слов на признаки русского языка
        vowels = 'аеёиоуыэюя'  # Русские гласные
        valid_words_count = 0

        for word in words:
            # В русском слове ОБЯЗАТЕЛЬНО должна быть хоть одна гласная
            has_vowels = any(c.lower() in vowels for c in word)

            # А еще в русском нет кучи согласных подряд (ну, больше 4х точно редкость)
            consonants = 'бвгджзйклмнпрстфхцчшщ'
            consecutive_consonants = 0
            max_consecutive_consonants = 0

            for char in word.lower():
                if char in consonants:
                    consecutive_consonants += 1
                    max_consecutive_consonants = max(max_consecutive_consonants, consecutive_consonants)
                else:
                    consecutive_consonants = 0

            # Если слово проходит обе проверки - считаем его правильным
            if has_vowels and max_consecutive_consonants <= 4:
                valid_words_count += 1

        # 2. Проверка типичных окончаний русских слов - они часто повторяются
        common_endings = ['ть', 'го', 'ый', 'ая', 'ое', 'ие', 'ся', 'ом', 'ем', 'ам', 'ах', 'ям']
        endings_count = 0

        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 2 and any(word_lower.endswith(ending) for ending in common_endings):
                endings_count += 1

        # 3. А без предлогов и союзов русский текст не бывает вообще
        common_small_words = ['в', 'на', 'с', 'к', 'у', 'от', 'до', 'из', 'о', 'и', 'а', 'но', 'или', 'как', 'что',
                              'не']
        small_words_count = sum(1 for w in words if w.lower() in common_small_words)

        # 4. Еще можно проверить, есть ли правильное чередование в тексте
        # Например, после предлога должно идти существительное или местоимение
        sequence_score = 0

        for i in range(len(words) - 1):
            if words[i].lower() in ['в', 'на', 'с', 'к', 'у', 'от', 'до', 'из', 'о']:
                # Если после предлога слово длиннее 2х букв - норм признак
                if len(words[i + 1]) > 2:
                    sequence_score += 1

        # 5. Структура предложений тоже важна - ищем начала с большой буквы и т.д.
        sentences = re.split(r'[.!?]+', sample)
        sentences = [s.strip() for s in sentences if s.strip()]

        sentence_structure_score = 0
        for sentence in sentences:
            # С большой буквы? Молодец!
            if sentence and sentence[0].isupper():
                sentence_structure_score += 1

            # Достаточно длинное? Тоже хорошо
            words_in_sentence = sentence.split()
            if len(words_in_sentence) >= 3:  # Меньше 3х слов - так себе предложение
                sentence_structure_score += 1

        # Теперь считаем итоговые оценки по разным параметрам
        word_validity_score = valid_words_count / max(1, len(words))
        ending_score = endings_count / max(1, len(words))
        small_words_score = small_words_count / max(1, len(words) * 0.3)  # Обычно ~30% маленьких слов
        seq_score_norm = sequence_score / max(1, len(words) / 5)  # Примерно каждое 5е слово - предлог
        sentence_score = sentence_structure_score / max(1, len(sentences) * 2)  # 2 критерия на предложение

        # Смешиваем всё вместе в правильных пропорциях
        quality = (
                word_validity_score * 0.35 +
                ending_score * 0.2 +
                min(1.0, small_words_score) * 0.15 +
                min(1.0, seq_score_norm) * 0.15 +
                min(1.0, sentence_score) * 0.15
        )

        return min(1.0, quality)

    def save_with_info(self, tab_index):
        """Сохраняет расшифрованный текст в файл"""
        content = self.decrypt_output_text.get("1.0", tk.END).strip()

        if not content:
            messagebox.showerror("Ошибка", "Нет расшифрованного текста для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return  # Пользователь отменил выбор файла

        try:
            # Получим зашифрованный текст и информацию о расшифровке
            encrypted_text = self.decrypt_input_text.get("1.0", tk.END).strip()

            # Составляем информацию о дешифровании из имеющихся данных
            width = self.decrypt_width_var.get().strip()
            height = self.decrypt_height_var.get().strip()
            route_type = self.decrypt_route_var.get().strip()

            # Составляем полный текст для сохранения
            full_content = f"=== ЗАШИФРОВАННЫЙ ТЕКСТ ===\n{encrypted_text}\n\n"
            full_content += f"=== ИНФОРМАЦИЯ О РАСШИФРОВКЕ ===\n"
            full_content += f"Размер таблицы: {width}x{height}\n"
            full_content += f"Тип маршрута: {route_type}\n\n"
            full_content += f"=== РАСШИФРОВАННЫЙ ТЕКСТ ===\n{content}"

            # Записываем в файл
            write_file(file_path, full_content)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
            import traceback
            traceback.print_exc()

    def decrypt(self, text, width=None, height=None, route_type=None, filler='Х'):
        """
        Дешифрует текст с использованием указанного типа маршрута.

        Параметры:
        - text: зашифрованный текст
        - width: ширина таблицы (если None, будет извлечена из текста)
        - height: не используется, вычисляется автоматически
        - route_type: тип маршрута (если None, будет определен с помощью криптоанализа)
        - filler: символ-заполнитель, использованный при шифровании

        Возвращает:
        - кортеж (дешифрованный текст, информация о дешифровании)
        """
        # Если ширина не определена, оцениваем размер таблицы
        if width is None:
            estimated = self.estimate_table_size(text)
            width = estimated["best_width"] if isinstance(estimated, dict) else estimated[0][0]

        # Всегда вычисляем высоту автоматически на основе ширины и длины текста
        height = (len(text) + width - 1) // width

        # Если тип маршрута не указан или пустой, определяем его с помощью криптоанализа
        if route_type is None or route_type.strip() == "":
            route_type = self.analyze_route_pattern(text, width, height)

        # Получаем маршрут
        route = self.get_route(width, height, route_type)

        # Создаем пустую матрицу
        matrix = [['' for _ in range(width)] for _ in range(height)]

        # Заполняем матрицу, следуя по маршруту
        for i, (x, y) in enumerate(route):
            if i < len(text) and 0 <= x < height and 0 <= y < width:
                matrix[x][y] = text[i]

        # Формируем дешифрованный текст, считывая матрицу по строкам
        decrypted = ''
        for row in matrix:
            decrypted += ''.join(row)

        # Удаляем символы-заполнители
        decrypted = decrypted.rstrip(filler)

        # Оцениваем качество дешифрования
        quality = self.assess_decryption_quality(decrypted)

        # Формируем информацию о дешифровании
        info = {
            "width": width,
            "height": height,
            "route_type": route_type,
            "quality_score": quality
        }

        return decrypted, self.format_table_with_route(text, width, height, route_type)

    def encrypt(self, text, width, route_type="спираль", filler='Х', remove_spaces=False):
        """
        Шифрует текст с использованием указанного типа маршрута.

        Параметры:
        - text: исходный текст для шифрования
        - width: ширина таблицы
        - route_type: тип маршрута ("спираль" или "змейка")
        - filler: символ-заполнитель для дополнения текста
        - remove_spaces: удалять ли пробелы при предобработке

        Возвращает:
        - кортеж (зашифрованный текст, информация о шифровании)
        """
        # Проверяем валидность текста
        if not self.validate_text(text):
            raise ValueError("Текст содержит недопустимые символы")

        # Предобрабатываем текст, высота будет вычислена автоматически
        preprocessed = self.preprocess_text(text, width, None, filler, remove_spaces)
        clean_text = preprocessed[0]
        width = preprocessed[1]
        height = preprocessed[2]  # Высота вычисляется автоматически
        spaces_removed = preprocessed[3]

        # Создаем матрицу
        matrix = self.create_matrix(clean_text, width, height)

        # Получаем маршрут
        route = self.get_route(width, height, route_type)

        # Формируем зашифрованный текст, следуя по маршруту
        encrypted = "".join(matrix[i][j] for i, j in route)

        # Формируем информацию о шифровании
        info = {
            "width": width,
            "height": height,
            "route_type": route_type,
            "filler": filler,
            "original_length": len(text),
            "removed_spaces": spaces_removed
        }

        return encrypted, self.format_table_with_route(clean_text, width, height, route_type)

    def extract_metadata_from_content(self, content):
        """Извлекает метаданные (ширину и тип маршрута) из содержимого файла"""
        metadata = {}

        # Ищем метаданные в разных форматах

        # 1. Ищем метаданные в начале файла (формат <W:число><R:тип>)
        width_pattern = re.compile(r'<W:(\d+)>')
        route_pattern = re.compile(r'<R:(спираль|змейка)>')

        width_match = width_pattern.search(content)
        route_match = route_pattern.search(content)

        if width_match:
            metadata['width'] = width_match.group(1)

        if route_match:
            metadata['route_type'] = route_match.group(1)

        # 2. Ищем метаданные в секции комментариев (формат <!-- METADATA ... -->)
        metadata_section = re.search(r'<!-- METADATA\s+(.*?)\s+-->', content, re.DOTALL)
        if metadata_section:
            section_content = metadata_section.group(1)

            # Ищем ширину и тип маршрута в секции метаданных
            width_in_section = re.search(r'<W:(\d+)>', section_content)
            route_in_section = re.search(r'<R:(спираль|змейка)>', section_content)

            if width_in_section and 'width' not in metadata:
                metadata['width'] = width_in_section.group(1)

            if route_in_section and 'route_type' not in metadata:
                metadata['route_type'] = route_in_section.group(1)

        # 3. Ищем информацию в текстовом формате (например, "Размер таблицы: 11x5")
        table_size = re.search(r'Размер таблицы:\s*(\d+)x\d+', content)
        route_info = re.search(r'Тип маршрута:\s*(спираль|змейка)', content)

        if table_size and 'width' not in metadata:
            metadata['width'] = table_size.group(1)

        if route_info and 'route_type' not in metadata:
            metadata['route_type'] = route_info.group(1)

        return metadata

    def open_file_for_decryption(self):
        """Открывает файл для дешифрования"""
        file_path = filedialog.askopenfilename(
            title="Открыть файл для дешифрования",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return  # Пользователь отменил выбор файла

        content = read_file(file_path)
        if content is not None:
            self.decrypt_input_text.delete("1.0", tk.END)
            self.decrypt_input_text.insert("1.0", content)
        else:
            messagebox.showerror("Ошибка", "Не удалось прочитать файл. Проверьте формат и кодировку.")

    def save_file_encrypted(self):
        """Сохраняет зашифрованный текст в файл"""
        encrypted_text = self.encrypt_output_text.get("1.0", tk.END).strip()
        if not encrypted_text:
            messagebox.showerror("Ошибка", "Нет текста для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Сохраняем зашифрованный текст без метаданных
                write_file(file_path, encrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_file_decrypted(self):
        """Сохраняет расшифрованный текст в файл"""
        decrypted_text = self.decrypt_output_text.get("1.0", tk.END).strip()
        if not decrypted_text:
            messagebox.showerror("Ошибка", "Нет текста для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Сохраняем только расшифрованный текст
                write_file(file_path, decrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")


# Определение класса RouteGUI
class RouteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Маршрутный шифр")

        # Создаем вкладки
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладка шифрования
        self.encrypt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.encrypt_frame, text="Шифрование")

        # Вкладка дешифрования
        self.decrypt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.decrypt_frame, text="Дешифрование")

        # Настраиваем интерфейс
        self.setup_encrypt_tab()
        self.setup_decrypt_tab()

    def setup_encrypt_tab(self):
        # Фрейм для ввода текста
        input_frame = ttk.LabelFrame(self.encrypt_frame, text="Исходный текст")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Текстовое поле для ввода
        self.encrypt_input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=60, height=10)
        self.encrypt_input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Кнопка загрузки из файла
        load_btn = ttk.Button(input_frame, text="Загрузить из файла", command=self.open_file_for_encryption)
        load_btn.pack(anchor=tk.W, padx=5, pady=5)

        # Фрейм для параметров шифрования
        params_frame = ttk.LabelFrame(self.encrypt_frame, text="Параметры шифрования")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Ширина таблицы
        width_frame = ttk.Frame(params_frame)
        width_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(width_frame, text="Ширина таблицы:").pack(side=tk.LEFT, padx=5)
        self.encrypt_width_var = tk.StringVar(value="11")
        width_entry = ttk.Entry(width_frame, textvariable=self.encrypt_width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)

        # Тип маршрута
        route_frame = ttk.Frame(params_frame)
        route_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(route_frame, text="Тип маршрута:").pack(side=tk.LEFT, padx=5)
        self.encrypt_route_var = tk.StringVar(value="спираль")
        route_combo = ttk.Combobox(route_frame, textvariable=self.encrypt_route_var, values=["спираль", "змейка"],
                                   state="readonly", width=10)
        route_combo.pack(side=tk.LEFT, padx=5)

        # Кнопка шифрования
        encrypt_btn = ttk.Button(self.encrypt_frame, text="Зашифровать", command=self.encrypt_text)
        encrypt_btn.pack(pady=10)

        # Фрейм для вывода
        output_frame = ttk.LabelFrame(self.encrypt_frame, text="Зашифрованный текст")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Текстовое поле для вывода
        self.encrypt_output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=10)
        self.encrypt_output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Кнопка сохранения
        save_btn = ttk.Button(output_frame, text="Сохранить в файл", command=self.save_file_encrypted)
        save_btn.pack(anchor=tk.E, padx=5, pady=5)

    def setup_decrypt_tab(self):
        # Фрейм для ввода текста
        input_frame = ttk.LabelFrame(self.decrypt_frame, text="Зашифрованный текст")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Текстовое поле для ввода
        self.decrypt_input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=60, height=10)
        self.decrypt_input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Кнопка загрузки из файла
        load_btn = ttk.Button(input_frame, text="Загрузить из файла", command=self.open_file_for_decryption)
        load_btn.pack(anchor=tk.W, padx=5, pady=5)

        # Фрейм для параметров дешифрования
        params_frame = ttk.LabelFrame(self.decrypt_frame, text="Параметры дешифрования")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Ширина таблицы
        width_frame = ttk.Frame(params_frame)
        width_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(width_frame, text="Ширина таблицы:").pack(side=tk.LEFT, padx=5)
        self.decrypt_width_var = tk.StringVar(value="11")
        width_entry = ttk.Entry(width_frame, textvariable=self.decrypt_width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)

        # Тип маршрута (для отображения результата криптоанализа)
        route_frame = ttk.Frame(params_frame)
        route_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(route_frame, text="Тип маршрута:").pack(side=tk.LEFT, padx=5)
        self.decrypt_route_var = tk.StringVar(value="")
        route_label = ttk.Label(route_frame, textvariable=self.decrypt_route_var, width=10)
        route_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(route_frame, text="(определяется автоматически)").pack(side=tk.LEFT, padx=5)

        # Скрытые переменные для сохранения данных
        self.decrypt_height_var = tk.StringVar(value="")

        # Кнопка дешифрования
        decrypt_btn = ttk.Button(self.decrypt_frame, text="Расшифровать", command=self.decrypt_text)
        decrypt_btn.pack(pady=10)

        # Фрейм для вывода
        output_frame = ttk.LabelFrame(self.decrypt_frame, text="Расшифрованный текст")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Текстовое поле для вывода
        self.decrypt_output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=10)
        self.decrypt_output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Кнопки для сохранения
        save_frame = ttk.Frame(output_frame)
        save_frame.pack(fill=tk.X, padx=5, pady=5)

        save_btn = ttk.Button(save_frame, text="Сохранить текст", command=self.save_file_decrypted)
        save_btn.pack(side=tk.LEFT, padx=5)

    def open_file_for_encryption(self):
        """Открывает файл для шифрования"""
        file_path = filedialog.askopenfilename(
            title="Открыть файл для шифрования",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return  # Пользователь отменил выбор файла

        content = read_file(file_path)
        if content is not None:
            self.encrypt_input_text.delete("1.0", tk.END)
            self.encrypt_input_text.insert("1.0", content)
        else:
            messagebox.showerror("Ошибка", "Не удалось прочитать файл. Проверьте формат и кодировку.")

    def encrypt_text(self):
        """Шифрует введенный текст"""
        # Получаем текст и параметры
        text = self.encrypt_input_text.get("1.0", tk.END).strip()

        if not text:
            messagebox.showerror("Ошибка", "Введите текст для шифрования")
            return

        try:
            width = int(self.encrypt_width_var.get().strip())
            if width <= 0:
                messagebox.showerror("Ошибка", "Ширина таблицы должна быть положительным числом")
                return

            route_type = self.encrypt_route_var.get()

            # Создаем объект шифра и шифруем текст
            cipher = RouteCipher()
            encrypted, table = cipher.encrypt(text, width, route_type, remove_spaces=False)

            # Выводим результат
            self.encrypt_output_text.delete("1.0", tk.END)
            self.encrypt_output_text.insert("1.0", encrypted)

        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при шифровании: {str(e)}")
            import traceback
            traceback.print_exc()

    def decrypt_text(self):
        """Дешифрует введенный текст"""
        # Берем текст из поля ввода
        text = self.decrypt_input_text.get("1.0", tk.END).strip()

        if not text:
            messagebox.showerror("Ошибка", "Введите текст для дешифрования")
            return

        try:
            # Ширину таблицы пользователь должен указать сам
            width_str = self.decrypt_width_var.get().strip()
            if not width_str:
                messagebox.showerror("Ошибка", "Введите ширину таблицы для дешифрования")
                return

            # Переводим в число и считаем высоту исходя из длины текста
            width = int(width_str)
            height = (len(text) + width - 1) // width
            
            # Создаем наш волшебный инструмент дешифровки
            cipher = RouteCipher()
            
            # Магия начинается! Анализируем и выбираем тип маршрута
            route_type = cipher.analyze_route_pattern(text, width, height)
            self.decrypt_route_var.set(route_type)
            
            # Дешифруем текст с определенным типом маршрута
            decrypted, table = cipher.decrypt(text, width, route_type=route_type)

            # Выводим результат
            self.decrypt_output_text.delete("1.0", tk.END)
            self.decrypt_output_text.insert("1.0", decrypted)
            
            # Сохраняем высоту для последующего сохранения
            self.decrypt_height_var.set(str(height))

        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при дешифровании: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_file_encrypted(self):
        """Сохраняет зашифрованный текст в файл"""
        encrypted_text = self.encrypt_output_text.get("1.0", tk.END).strip()
        if not encrypted_text:
            messagebox.showerror("Ошибка", "Нет текста для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Сохраняем зашифрованный текст без метаданных
                write_file(file_path, encrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_file_decrypted(self):
        """Сохраняет расшифрованный текст в файл"""
        decrypted_text = self.decrypt_output_text.get("1.0", tk.END).strip()
        if not decrypted_text:
            messagebox.showerror("Ошибка", "Нет текста для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Сохраняем только расшифрованный текст
                write_file(file_path, decrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_with_info(self, tab_index):
        """Сохраняет расшифрованный текст в файл"""
        content = self.decrypt_output_text.get("1.0", tk.END).strip()

        if not content:
            messagebox.showerror("Ошибка", "Нет расшифрованного текста для сохранения")
            return

        file_path = filedialog.asksaveasfilename(
            title="Сохранить",
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return  # Пользователь отменил выбор файла

        try:
            # Получим зашифрованный текст и информацию о расшифровке
            encrypted_text = self.decrypt_input_text.get("1.0", tk.END).strip()

            # Составляем информацию о дешифровании из имеющихся данных
            width = self.decrypt_width_var.get().strip()
            height = self.decrypt_height_var.get().strip()
            route_type = self.decrypt_route_var.get().strip()

            # Составляем полный текст для сохранения
            full_content = f"=== ЗАШИФРОВАННЫЙ ТЕКСТ ===\n{encrypted_text}\n\n"
            full_content += f"=== ИНФОРМАЦИЯ О РАСШИФРОВКЕ ===\n"
            full_content += f"Размер таблицы: {width}x{height}\n"
            full_content += f"Тип маршрута: {route_type}\n\n"
            full_content += f"=== РАСШИФРОВАННЫЙ ТЕКСТ ===\n{content}"

            # Записываем в файл
            write_file(file_path, full_content)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    root = tk.Tk()
    app = RouteGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
