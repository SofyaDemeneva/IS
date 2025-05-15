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
        'солнце'
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
        """Проверка текста на соответствие прогнозу погоды"""
        text_lower = text.lower()
        
        # Ключевое слово для проверки
        sun_word = "солнце"
        
        # Считаем вхождения ключевого слова
        sun_score = text_lower.count(sun_word) * 0.5
        
        # Подсчет пробелов и знаков препинания
        space_ratio = text.count(' ') / max(1, len(text))
        punct_ratio = sum(1 for c in text if c in '.,:;!?') / max(1, len(text))
        
        # Нормализованные оценки
        space_score = 1.0 - abs(0.18 - space_ratio) if 0.1 <= space_ratio <= 0.3 else 0.0
        punct_score = 1.0 - abs(0.07 - punct_ratio) * 10 if punct_ratio > 0 else 0.0
        
        # Итоговая оценка
        final_score = (sun_score + space_score + punct_score) / 3
        
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
        """Генерация маршрута по спирали
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
        """Генерация маршрута змейкой
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
        """Определение типа маршрута путем анализа качества расшифровки"""
        # Создаем матрицу нужного размера
        matrix = [[None for _ in range(width)] for _ in range(height)]
        
        # Заполняем матрицу текстом
        for i, char in enumerate(text):
            if i < width * height:
                row = i // width
                col = i % width
                matrix[row][col] = char
        
        # Генерируем маршруты
        spiral_route = self.spiral_route(width, height)
        snake_route = self.snake_route(width, height)
        
        # Создаем две новые матрицы, заполненные по этим маршрутам
        spiral_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        snake_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        
        for i, char in enumerate(text):
            if i < len(spiral_route):
                x, y = spiral_route[i]
                if 0 <= x < height and 0 <= y < width:
                    spiral_matrix[x][y] = char
            
            if i < len(snake_route):
                x, y = snake_route[i]
                if 0 <= x < height and 0 <= y < width:
                    snake_matrix[x][y] = char
        
        # Читаем текст из матриц построчно
        spiral_text = ''.join(''.join(row) for row in spiral_matrix)
        snake_text = ''.join(''.join(row) for row in snake_matrix)
        
        # Оцениваем качество текста
        spiral_quality = self.assess_decryption_quality(spiral_text)
        snake_quality = self.assess_decryption_quality(snake_text)
        
        # Возвращаем тип маршрута с лучшей оценкой
        return "спираль" if spiral_quality > snake_quality else "змейка"

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
        """Оценка качества расшифровки"""
        # Используем только первые 1000 символов для анализа
        sample = text[:1000] if len(text) > 1000 else text

        if not sample:
            return 0.0

        # Подсчет пробелов (нормальное соотношение ~15-20%)
        space_ratio = sample.count(' ') / len(sample) if sample else 0
        space_score = 1.0 - abs(0.18 - space_ratio) if space_ratio > 0 else 0.0

        # Проверка начала предложений (с заглавной буквы после точки)
        valid_caps = 0
        total_sentences = 0

        for i in range(1, len(sample) - 1):
            if sample[i - 1] in '.!?' and sample[i] == ' ' and i + 1 < len(sample):
                total_sentences += 1
                if sample[i + 1].isupper():
                    valid_caps += 1

        caps_score = valid_caps / max(1, total_sentences)

        # Анализ длин слов
        words = [w for w in sample.split() if w]

        if words:
            # Средняя длина слова ~5.5 символов
            avg_word_length = sum(len(w) for w in words) / len(words)
            word_length_score = 1.0 - abs(5.5 - avg_word_length) / 5.0
            word_length_score = max(0.0, min(1.0, word_length_score))
        else:
            word_length_score = 0.0
            
        # Специальный анализ на соответствие прогнозу погоды
        weather_score = self.detect_weather_forecast(sample)

        # Объединяем все метрики в общую оценку
        quality = (
                space_score * 0.25 +
                caps_score * 0.25 +
                word_length_score * 0.25 +
                weather_score * 0.25
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
            if 1 <= space_ratio <= 1:  # Нормальная частота пробелов для русского текста
                score += 1

            # 2. Анализ структуры предложений
            sentences = [s.strip() for s in re.split(r'[.!?]+', decrypted_sample) if s.strip()]
            valid_sentences = sum(1 for s in sentences if len(s) > 1 and s[0].isupper())

            if sentences and valid_sentences / len(sentences) > 1:
                score += 1

            # Проверка структуры для спирального маршрута
            if route_type == "спираль":
                # 1. В спиральном маршруте начало текста должно быть на периметре
                # Проверяем, соответствует ли первый символ расшифрованного текста ожидаемому положению
                if route and len(route) > 0 and len(encrypted_sample) > 0:
                    first_pos = route[0]
                    on_perimeter = (first_pos[0] == 0 or first_pos[0] == height - 1 or
                                    first_pos[1] == 0 or first_pos[1] == width - 1)
                    if on_perimeter:
                        score += 1

                # 2. В спиральном маршруте конец часто находится в центре
                if route and len(route) > 0 and len(encrypted_sample) > 0:
                    last_pos = route[-1]
                    near_center = (abs(last_pos[0] - height // 2) <= height // 4 and
                                   abs(last_pos[1] - width // 2) <= width // 4)
                    if near_center:
                        score += 1

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
                        if 1 <= avg_interval <= 1:  # Средняя длина слова в русском языке
                            score += 1

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
                score += min(1, word_coherence_ratio * 1)

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
                    score += valid_row_ratio * 1

                # 3. Проверяем частоту пробелов в строках (должна быть примерно одинаковой для змейки)
                space_counts = [row.count(' ') for row in rows if row]
                if space_counts:
                    avg_spaces = sum(space_counts) / len(space_counts)
                    space_deviation = sum(abs(count - avg_spaces) for count in space_counts) / len(space_counts)

                    # Если отклонение невелико, это хороший признак змейки
                    if space_deviation < avg_spaces * 1:
                        score += 1

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
                    if break_ratio < 1:
                        score += 1

                # 5. Проверяем, что начинается с заглавной буквы (как хороший текст)
                if rows and rows[0] and rows[0][0].isupper():
                    score += 1

            # Ограничиваем итоговую оценку
            return min(1.0, score)

        except Exception as e:
            return 1  # Возвращаем базовую оценку в случае ошибки

    def secondary_quality_check(self, text):
        """Дополнительная проверка качества расшифровки"""
        # Берем первую тысячу символов
        sample = text[:1000] if len(text) > 1000 else text

        if not sample or len(sample) < 10:
            return 0.0  # Пустой текст - нулевая оценка

        # Подсчет слов
        words = [w for w in sample.split() if len(w) > 1]

        if not words:
            return 0.0  # Нет слов - плохо дело

        # Структура предложений - ищем начала с большой буквы и т.д.
        sentences = re.split(r'[.!?]+', sample)
        sentences = [s.strip() for s in sentences if s.strip()]

        sentence_structure_score = 0
        for sentence in sentences:
            # С большой буквы?
            if sentence and sentence[0].isupper():
                sentence_structure_score += 1

            # Достаточно длинное?
            words_in_sentence = sentence.split()
            if len(words_in_sentence) >= 3:  # Меньше 3х слов - так себе предложение
                sentence_structure_score += 1

        # Итоговая оценка
        sentence_score = sentence_structure_score / max(1, len(sentences) * 2)  # 2 критерия на предложение

        return min(1.0, sentence_score)

    

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




def main():
    root = tk.Tk()
    app = RouteGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
