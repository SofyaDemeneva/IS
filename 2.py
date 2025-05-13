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
        text_lower = text.lower()
        
        popular_bigrams = ['ст', 'но', 'то', 'на', 'ен', 'ов', 'ни', 'ра', 'во', 
                          'ко', 'ал', 'ли', 'по', 'ре', 'ол', 'пр', 'ть', 'ат', 
                          'ет', 'та', 'го', 'ос', 'ер', 'ит', 'нн', 'ск', 'ны', 'ие']
        
        popular_trigrams = ['ост', 'ого', 'ени', 'ста', 'про', 'ная', 'ени', 'что', 
                           'тор', 'ать', 'кот', 'его', 'ном', 'ого', 'ова', 'ств']
        
        sun_word = "солнце"
        sun_ngrams = []
        
        for i in range(len(sun_word)-1):
            sun_ngrams.append(sun_word[i:i+2])
            
        for i in range(len(sun_word)-2):
            sun_ngrams.append(sun_word[i:i+3])
            
        for i in range(len(sun_word)-3):
            sun_ngrams.append(sun_word[i:i+4])
        for i in range(len(sun_word)-4):
            sun_ngrams.append(sun_word[i:i+5])
        
        bigram_score = 0
        for bigram in popular_bigrams:
            count = text_lower.count(bigram)
            bigram_score += count
        
        total_chars = max(1, len(text_lower) - 1)
        bigram_ratio = min(1.0, bigram_score / (total_chars * 0.3))
        
        trigram_score = 0
        for trigram in popular_trigrams:
            count = text_lower.count(trigram)
            trigram_score += count * 1.5
        
        trigram_ratio = min(1.0, trigram_score / (total_chars * 0.2))
        
        sun_score = 0
        for ngram in sun_ngrams:
            ngram_count = text_lower.count(ngram)
            if ngram_count > 0:
                sun_score += ngram_count * (len(ngram) / 2.0)
        
        if sun_word in text_lower:
            sun_score += 5.0
            
        sun_score = min(1.0, sun_score / 15.0)
        
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
        matrix = []
        for i in range(height):
            row = list(text[i * width: (i + 1) * width])
            matrix.append(row)
        return matrix

    def spiral_route(self, width, height):
        matrix = [[0] * width for _ in range(height)]
        x, y = 0, 0
        dx, dy = 0, 1
        route = []

        for _ in range(width * height):
            route.append((x, y))
            matrix[x][y] = 1

            next_x, next_y = x + dx, y + dy

            if (next_x < 0 or next_x >= height or
                    next_y < 0 or next_y >= width or
                    matrix[next_x][next_y] == 1):
                dx, dy = dy, -dx
                next_x, next_y = x + dx, y + dy

            x, y = next_x, next_y

        return route

    def snake_route(self, width, height):
        route = []
        for i in range(height):
            if i % 2 == 0:
                for j in range(width):
                    route.append((i, j))
            else:
                for j in range(width - 1, -1, -1):
                    route.append((i, j))
        return route

    def diagonal_route(self, width, height):
        route = []
        for sum_idx in range(width + height - 1):
            diagonal = []
            for i in range(sum_idx + 1):
                j = sum_idx - i
                if i < height and j < width:
                    diagonal.append((i, j))
            
            if sum_idx % 2 == 0:
                route.extend(diagonal)
            else:
                route.extend(reversed(diagonal))
        
        return route

    def spiral_counterclockwise_route(self, width, height):
        matrix = [[0] * width for _ in range(height)]
        x, y = 0, 0
        dx, dy = 1, 0
        route = []

        for _ in range(width * height):
            route.append((x, y))
            matrix[x][y] = 1

            next_x, next_y = x + dx, y + dy

            if (next_x < 0 or next_x >= height or
                    next_y < 0 or next_y >= width or
                    matrix[next_x][next_y] == 1):
                dx, dy = -dy, dx
                next_x, next_y = x + dx, y + dy

            x, y = next_x, next_y

        return route

    def zigzag_vertical_route(self, width, height):
        route = []
        for j in range(width):
            if j % 2 == 0:
                for i in range(height):
                    route.append((i, j))
            else:
                for i in range(height - 1, -1, -1):
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
        if not text or width <= 0 or height <= 0:
            return "спираль"
        
        if len(text) < width * 2:
            return "спираль"  

        spiral_route = self.spiral_route(width, height)
        snake_route = self.snake_route(width, height)
        diagonal_route = self.diagonal_route(width, height)
        spiral_counter_route = self.spiral_counterclockwise_route(width, height)
        zigzag_vert_route = self.zigzag_vertical_route(width, height)

        spiral_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        snake_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        diagonal_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        spiral_counter_matrix = [[' ' for _ in range(width)] for _ in range(height)]
        zigzag_vert_matrix = [[' ' for _ in range(width)] for _ in range(height)]

        text_normalized = text[:min(len(text), width * height)]

        for idx, char in enumerate(text_normalized):
            if idx < len(spiral_route):
                i, j = spiral_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    spiral_matrix[i][j] = char

            if idx < len(snake_route):
                i, j = snake_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    snake_matrix[i][j] = char
                    
            if idx < len(diagonal_route):
                i, j = diagonal_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    diagonal_matrix[i][j] = char
                    
            if idx < len(spiral_counter_route):
                i, j = spiral_counter_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    spiral_counter_matrix[i][j] = char
                    
            if idx < len(zigzag_vert_route):
                i, j = zigzag_vert_route[idx]
                if 0 <= i < height and 0 <= j < width:
                    zigzag_vert_matrix[i][j] = char

        spiral_text = ''.join(''.join(row) for row in spiral_matrix)
        snake_text = ''.join(''.join(row) for row in snake_matrix)
        diagonal_text = ''.join(''.join(row) for row in diagonal_matrix)
        spiral_counter_text = ''.join(''.join(row) for row in spiral_counter_matrix)
        zigzag_vert_text = ''.join(''.join(row) for row in zigzag_vert_matrix)

        spiral_quality = self.assess_decryption_quality(spiral_text)
        snake_quality = self.assess_decryption_quality(snake_text)
        diagonal_quality = self.assess_decryption_quality(diagonal_text)
        spiral_counter_quality = self.assess_decryption_quality(spiral_counter_text)
        zigzag_vert_quality = self.assess_decryption_quality(zigzag_vert_text)
        
        spiral_linguistic = self.secondary_quality_check(spiral_text)
        snake_linguistic = self.secondary_quality_check(snake_text)
        diagonal_linguistic = self.secondary_quality_check(diagonal_text)
        spiral_counter_linguistic = self.secondary_quality_check(spiral_counter_text)
        zigzag_vert_linguistic = self.secondary_quality_check(zigzag_vert_text)
        
        spiral_ngram_score = self.detect_weather_forecast(spiral_text)
        snake_ngram_score = self.detect_weather_forecast(snake_text)
        diagonal_ngram_score = self.detect_weather_forecast(diagonal_text)
        spiral_counter_ngram_score = self.detect_weather_forecast(spiral_counter_text)
        zigzag_vert_ngram_score = self.detect_weather_forecast(zigzag_vert_text)
        
        spiral_score = (spiral_quality * 0.4 + 
                       spiral_linguistic * 0.3 + 
                       spiral_ngram_score * 0.3)
                       
        snake_score = (snake_quality * 0.4 + 
                      snake_linguistic * 0.3 + 
                      snake_ngram_score * 0.3)
                      
        diagonal_score = (diagonal_quality * 0.4 + 
                         diagonal_linguistic * 0.3 + 
                         diagonal_ngram_score * 0.3)
                         
        spiral_counter_score = (spiral_counter_quality * 0.4 + 
                               spiral_counter_linguistic * 0.3 + 
                               spiral_counter_ngram_score * 0.3)
                               
        zigzag_vert_score = (zigzag_vert_quality * 0.4 + 
                            zigzag_vert_linguistic * 0.3 + 
                            zigzag_vert_ngram_score * 0.3)
        
        if abs(width - height) <= 2:
            spiral_score *= 1.1
            spiral_counter_score *= 1.1
            
        if width > height * 2:
            snake_score *= 1.15
            
        if height > width * 2:
            zigzag_vert_score *= 1.15
            
        if width == 11:
            spiral_score *= 1.1
            
        if width <= 5 and height <= 5:
            spiral_score *= 1.1
            spiral_counter_score *= 1.1
        
        scores = {
            "спираль": spiral_score,
            "змейка": snake_score,
            "диагональ": diagonal_score,
            "спираль_против": spiral_counter_score,
            "зигзаг_верт": zigzag_vert_score
        }
        
        best_route_type = max(scores, key=scores.get)
        return best_route_type

    def get_route(self, width, height, route_type):
        if route_type == "спираль":
            return self.spiral_route(width, height)
        elif route_type == "змейка":
            return self.snake_route(width, height)
        elif route_type == "диагональ":
            return self.diagonal_route(width, height)
        elif route_type == "спираль_против":
            return self.spiral_counterclockwise_route(width, height)
        elif route_type == "зигзаг_верт":
            return self.zigzag_vertical_route(width, height)
        else:
            raise ValueError(f"Неизвестный тип маршрута: {route_type}. Поддерживаемые типы: 'спираль', 'змейка', 'диагональ', 'спираль_против', 'зигзаг_верт'.")

    def format_table(self, matrix):
        if not matrix:
            return "Пустая таблица"

        if not isinstance(matrix, list) or len(matrix) == 0:
            return "Некорректная матрица"

        if not matrix[0] or not isinstance(matrix[0], list):
            return "Пустая или некорректная матрица"

        result = []
        for row in matrix:
            if not isinstance(row, list):
                continue
            result.append(''.join(str(cell) if cell is not None else ' ' for cell in row))

        if not result:
            return "Пустая таблица"

        return '\n'.join(result)

    def format_table_with_route(self, text, width, height, route_type="спираль"):
        if not text:
            return "Пустая таблица"

        if width <= 0 or height <= 0:
            return "Некорректные размеры таблицы"

        matrix = [[' ' for _ in range(width)] for _ in range(height)]

        try:
            route = self.get_route(width, height, route_type)

            if not route:
                return f"Ошибка: Не удалось создать маршрут типа '{route_type}' для таблицы {width}x{height}"

            for i, (x, y) in enumerate(route):
                if i < len(text) and 0 <= x < height and 0 <= y < width:
                    matrix[x][y] = text[i]

            return self.format_table(matrix)
        except Exception as e:
            return f"Ошибка при форматировании таблицы: {e}"

    def preprocess_text(self, text, width=None, height=None, filler='Х', remove_spaces=False):
        if not text:
            raise ValueError("Текст не может быть пустым")

        text = str(text)

        if text.startswith('\ufeff'):
            text = text[1:]

        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('«', '"').replace('»', '"')
        text = text.replace(chr(8239), ' ').replace(chr(8201), ' ')
        text = text.replace('\xa0', ' ')

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

        if width is None:
            return cleaned_text, 0, 0, spaces_removed

        if width <= 0:
            estimated = self.estimate_table_size(cleaned_text)
            width = estimated["best_width"] if isinstance(estimated, dict) else estimated[0][0]

        height = (len(cleaned_text) + width - 1) // width

        required_length = width * height

        if len(cleaned_text) < required_length:
            padding_length = required_length - len(cleaned_text)
            cleaned_text += filler * padding_length

        return cleaned_text, width, height, spaces_removed

    def get_fillers(self):
        fillers = []
        for char in 'ХЬЪЫЭЮЯ':
            fillers.append(char)
        return fillers

    def assess_decryption_quality(self, text):
        sample = text[:1000] if len(text) > 1000 else text

        if not sample:
            return 0.0

        russian_chars = sum(1 for c in sample if c.lower() in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        russian_ratio = russian_chars / len(sample) if sample else 0

        if russian_ratio < 0.3:
            return russian_ratio * 0.5

        space_ratio = sample.count(' ') / len(sample) if sample else 0
        space_score = 1.0 - 2.0 * abs(0.18 - space_ratio) if space_ratio > 0 else 0.0

        if space_ratio < 0.05 or space_ratio > 0.3:
            space_score = space_score / 2

        common_bigrams = ['ст', 'но', 'то', 'на', 'ен', 'ов', 'ни', 'ра', 'во', 'ко', 'ал', 'ли', 'по', 'ре', 'ол',
                          'пр', 'ть', 'ат', 'ет', 'та', 'го', 'ос', 'ер', 'ит', 'нн', 'ск', 'ны', 'ие']
        sample_lower = sample.lower()
        bigram_count = 0

        for bigram in common_bigrams:
            bigram_count += sample_lower.count(bigram)

        bigram_ratio = bigram_count / max(1, len(sample) - 1) * (10 / len(common_bigrams))

        punct_marks = '.,:;!?'
        punct_count = sum(1 for c in sample if c in punct_marks)
        punct_ratio = punct_count / len(sample) if sample else 0

        punct_score = 1.0 - abs(0.07 - punct_ratio) * 5.0 if punct_ratio > 0 else 0.0
        punct_score = max(0.0, min(1.0, punct_score))

        valid_caps = 0
        total_sentences = 0

        for i in range(1, len(sample) - 1):
            if sample[i - 1] in '.!?' and sample[i] == ' ' and i + 1 < len(sample):
                total_sentences += 1
                if sample[i + 1].isupper():
                    valid_caps += 1

        caps_score = valid_caps / max(1, total_sentences)

        vowels = 'аеёиоуыэюя'
        consonants = 'бвгджзйклмнпрстфхцчшщ'

        vowel_count = sum(1 for c in sample_lower if c in vowels)
        consonant_count = sum(1 for c in sample_lower if c in consonants)

        if vowel_count + consonant_count > 0:
            vowel_ratio = vowel_count / (vowel_count + consonant_count)
            vowel_consonant_score = 1.0 - abs(0.42 - vowel_ratio) * 2.5
            vowel_consonant_score = max(0.0, min(1.0, vowel_consonant_score))
        else:
            vowel_consonant_score = 0.0

        words = [w for w in sample.split() if w]

        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            word_length_score = 1.0 - abs(5.5 - avg_word_length) / 5.0
            word_length_score = max(0.0, min(1.0, word_length_score))

            long_words_ratio = sum(1 for w in words if len(w) > 15) / len(words)
            if long_words_ratio > 0.1:
                word_length_score *= (1.0 - long_words_ratio)
        else:
            word_length_score = 0.0
            
        weather_score = self.detect_weather_forecast(sample)

        quality = (
                russian_ratio * 0.2 +
                space_score * 0.15 +
                bigram_ratio * 0.1 +
                punct_score * 0.05 +
                caps_score * 0.1 +
                vowel_consonant_score * 0.1 +
                word_length_score * 0.1 +
                weather_score * 0.2
        )

        return min(1.0, max(0.0, quality))

    def analyze_route_structure(self, encrypted_text, decrypted_text, width, height, route_type):
        try:
            encrypted_sample = encrypted_text[:min(1000, len(encrypted_text))]
            decrypted_sample = decrypted_text[:min(1000, len(decrypted_text))]

            score = 0.5

            route = self.get_route(width, height, route_type)

            if not route:
                return score

            matrix = [[' ' for _ in range(width)] for _ in range(height)]
            for i, (x, y) in enumerate(route):
                if i < len(encrypted_sample) and 0 <= x < height and 0 <= y < width:
                    matrix[x][y] = encrypted_sample[i]

            space_ratio = decrypted_sample.count(' ') / max(1, len(decrypted_sample))
            if 0.15 <= space_ratio <= 0.25:
                score += 0.05

            sentences = [s.strip() for s in re.split(r'[.!?]+', decrypted_sample) if s.strip()]
            valid_sentences = sum(1 for s in sentences if len(s) > 5 and s[0].isupper())

            if sentences and valid_sentences / len(sentences) > 0.7:
                score += 0.05

            if route_type == "спираль":
                if route and len(route) > 0 and len(encrypted_sample) > 0:
                    first_pos = route[0]
                    on_perimeter = (first_pos[0] == 0 or first_pos[0] == height - 1 or
                                    first_pos[1] == 0 or first_pos[1] == width - 1)
                    if on_perimeter:
                        score += 0.1

                if route and len(route) > 0 and len(encrypted_sample) > 0:
                    last_pos = route[-1]
                    near_center = (abs(last_pos[0] - height // 2) <= height // 4 and
                                   abs(last_pos[1] - width // 2) <= width // 4)
                    if near_center:
                        score += 0.1

                edge_chars = []
                if height > 0 and width > 0:
                    edge_chars.extend([matrix[0][j] for j in range(min(width, len(matrix[0])))])
                if height > 0 and width > 0:
                    edge_chars.extend([matrix[i][width - 1] for i in range(1, min(height, len(matrix))) if
                                       width - 1 < len(matrix[i])])
                if height > 1 and width > 1:
                    if height - 1 < len(matrix):
                        edge_chars.extend(
                            [matrix[height - 1][j] for j in range(width - 2, -1, -1) if j < len(matrix[height - 1])])
                if height > 2 and width > 0:
                    edge_chars.extend([matrix[i][0] for i in range(height - 2, 0, -1) if 0 < len(matrix[i])])

                edge_text = ''.join(edge_chars)

                if edge_text:
                    space_positions = [i for i, c in enumerate(edge_text) if c == ' ']
                    if space_positions:
                        space_intervals = [space_positions[i + 1] - space_positions[i] for i in
                                           range(len(space_positions) - 1)]
                        avg_interval = sum(space_intervals) / max(1, len(space_intervals))
                        if 4 <= avg_interval <= 8:
                            score += 0.1

                word_coherence = 0
                for i in range(min(len(encrypted_sample), len(route)) - 1):
                    if i + 1 >= len(route):
                        continue

                    x1, y1 = route[i]
                    x2, y2 = route[i + 1]

                    if (0 <= x1 < height and 0 <= y1 < width and 0 <= x2 < height and 0 <= y2 < width and
                            matrix[x1][y1] != ' ' and matrix[x2][y2] != ' '):

                        if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1:
                            word_coherence += 1

                word_coherence_ratio = word_coherence / max(1, len(encrypted_sample) - 1)
                score += min(0.15, word_coherence_ratio * 0.5)

            elif route_type == "змейка":
                rows = []
                for i in range(min(height, len(matrix))):
                    row_indices = [(i, j) for j in range(width)] if i % 2 == 0 else [(i, j) for j in
                                                                                     range(width - 1, -1, -1)]
                    row_text = ''.join(
                        matrix[x][y] if x < height and y < width and y < len(matrix[x]) else ' ' for x, y in
                        row_indices)
                    rows.append(row_text.strip())

                valid_row_ends = 0
                for i, row in enumerate(rows):
                    if not row:
                        continue

                    if i % 2 == 0:
                        if row[-1] in ' .,!?:;':
                            valid_row_ends += 1
                    else:
                        if row[0] in ' .,!?:;':
                            valid_row_ends += 1

                if rows:
                    valid_row_ratio = valid_row_ends / len(rows)
                    score += valid_row_ratio * 0.15

                space_counts = [row.count(' ') for row in rows if row]
                if space_counts:
                    avg_spaces = sum(space_counts) / len(space_counts)
                    space_deviation = sum(abs(count - avg_spaces) for count in space_counts) / len(space_counts)

                    if space_deviation < avg_spaces * 0.3:
                        score += 0.1

                if len(rows) >= 2:
                    continuous_breaks = 0
                    for i in range(len(rows) - 1):
                        if not rows[i] or not rows[i + 1]:
                            continue

                        if i % 2 == 0:
                            if rows[i][-1].isalpha() and rows[i + 1][0].isalpha():
                                continuous_breaks += 1
                        else:
                            if rows[i][0].isalpha() and rows[i + 1][-1].isalpha():
                                continuous_breaks += 1

                    break_ratio = continuous_breaks / (len(rows) - 1)
                    if break_ratio < 0.3:
                        score += 0.15

                if rows and rows[0] and rows[0][0].isupper():
                    score += 0.05

            return min(1.0, score)

        except Exception as e:
            return 0.5

    def secondary_quality_check(self, text):
        sample = text[:1000] if len(text) > 1000 else text

        if not sample or len(sample) < 10:
            return 0.0

        words = [w for w in sample.split() if len(w) > 1]

        if not words:
            return 0.0

        vowels = 'аеёиоуыэюя'
        valid_words_count = 0

        for word in words:
            has_vowels = any(c.lower() in vowels for c in word)

            consonants = 'бвгджзйклмнпрстфхцчшщ'
            consecutive_consonants = 0
            max_consecutive_consonants = 0

            for char in word.lower():
                if char in consonants:
                    consecutive_consonants += 1
                    max_consecutive_consonants = max(max_consecutive_consonants, consecutive_consonants)
                else:
                    consecutive_consonants = 0

            if has_vowels and max_consecutive_consonants <= 4:
                valid_words_count += 1

        common_endings = ['ть', 'го', 'ый', 'ая', 'ое', 'ие', 'ся', 'ом', 'ем', 'ам', 'ах', 'ям']
        endings_count = 0

        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 2 and any(word_lower.endswith(ending) for ending in common_endings):
                endings_count += 1

        common_small_words = ['в', 'на', 'с', 'к', 'у', 'от', 'до', 'из', 'о', 'и', 'а', 'но', 'или', 'как', 'что',
                              'не']
        small_words_count = sum(1 for w in words if w.lower() in common_small_words)

        sequence_score = 0

        for i in range(len(words) - 1):
            if words[i].lower() in ['в', 'на', 'с', 'к', 'у', 'от', 'до', 'из', 'о']:
                if len(words[i + 1]) > 2:
                    sequence_score += 1

        sentences = re.split(r'[.!?]+', sample)
        sentences = [s.strip() for s in sentences if s.strip()]

        sentence_structure_score = 0
        for sentence in sentences:
            if sentence and sentence[0].isupper():
                sentence_structure_score += 1

            words_in_sentence = sentence.split()
            if len(words_in_sentence) >= 3:
                sentence_structure_score += 1

        word_validity_score = valid_words_count / max(1, len(words))
        ending_score = endings_count / max(1, len(words))
        small_words_score = small_words_count / max(1, len(words) * 0.3)
        seq_score_norm = sequence_score / max(1, len(words) / 5)
        sentence_score = sentence_structure_score / max(1, len(sentences) * 2)

        quality = (
                word_validity_score * 0.35 +
                ending_score * 0.2 +
                min(1.0, small_words_score) * 0.15 +
                min(1.0, seq_score_norm) * 0.15 +
                min(1.0, sentence_score) * 0.15
        )

        return min(1.0, quality)

    def save_with_info(self, tab_index):
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
            return

        try:
            encrypted_text = self.decrypt_input_text.get("1.0", tk.END).strip()

            width = self.decrypt_width_var.get().strip()
            height = self.decrypt_height_var.get().strip()
            route_type = self.decrypt_route_var.get().strip()

            full_content = f"=== ЗАШИФРОВАННЫЙ ТЕКСТ ===\n{encrypted_text}\n\n"
            full_content += f"=== ИНФОРМАЦИЯ О РАСШИФРОВКЕ ===\n"
            full_content += f"Размер таблицы: {width}x{height}\n"
            full_content += f"Тип маршрута: {route_type}\n\n"
            full_content += f"=== РАСШИФРОВАННЫЙ ТЕКСТ ===\n{content}"

            write_file(file_path, full_content)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")
            import traceback
            traceback.print_exc()

    def decrypt(self, text, width=None, height=None, route_type=None, filler='Х'):
        if width is None:
            estimated = self.estimate_table_size(text)
            width = estimated["best_width"] if isinstance(estimated, dict) else estimated[0][0]

        height = (len(text) + width - 1) // width

        if route_type is None or route_type.strip() == "":
            route_type = self.analyze_route_pattern(text, width, height)

        route = self.get_route(width, height, route_type)

        matrix = [['' for _ in range(width)] for _ in range(height)]

        for i, (x, y) in enumerate(route):
            if i < len(text) and 0 <= x < height and 0 <= y < width:
                matrix[x][y] = text[i]

        decrypted = ''
        for row in matrix:
            decrypted += ''.join(row)

        decrypted = decrypted.rstrip(filler)

        quality = self.assess_decryption_quality(decrypted)

        info = {
            "width": width,
            "height": height,
            "route_type": route_type,
            "quality_score": quality
        }

        return decrypted, self.format_table_with_route(text, width, height, route_type)

    def encrypt(self, text, width, route_type="спираль", filler='Х', remove_spaces=False):
        if not self.validate_text(text):
            raise ValueError("Текст содержит недопустимые символы")

        preprocessed = self.preprocess_text(text, width, None, filler, remove_spaces)
        clean_text = preprocessed[0]
        width = preprocessed[1]
        height = preprocessed[2]
        spaces_removed = preprocessed[3]

        matrix = self.create_matrix(clean_text, width, height)

        route = self.get_route(width, height, route_type)

        encrypted = "".join(matrix[i][j] for i, j in route)

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
        metadata = {}

        width_pattern = re.compile(r'<W:(\d+)>')
        route_pattern = re.compile(r'<R:(спираль|змейка)>')

        width_match = width_pattern.search(content)
        route_match = route_pattern.search(content)

        if width_match:
            metadata['width'] = width_match.group(1)

        if route_match:
            metadata['route_type'] = route_match.group(1)

        metadata_section = re.search(r'<!-- METADATA\s+(.*?)\s+-->', content, re.DOTALL)
        if metadata_section:
            section_content = metadata_section.group(1)

            width_in_section = re.search(r'<W:(\d+)>', section_content)
            route_in_section = re.search(r'<R:(спираль|змейка)>', section_content)

            if width_in_section and 'width' not in metadata:
                metadata['width'] = width_in_section.group(1)

            if route_in_section and 'route_type' not in metadata:
                metadata['route_type'] = route_in_section.group(1)

        table_size = re.search(r'Размер таблицы:\s*(\d+)x\d+', content)
        route_info = re.search(r'Тип маршрута:\s*(спираль|змейка)', content)

        if table_size and 'width' not in metadata:
            metadata['width'] = table_size.group(1)

        if route_info and 'route_type' not in metadata:
            metadata['route_type'] = route_info.group(1)

        return metadata

    def open_file_for_decryption(self):
        file_path = filedialog.askopenfilename(
            title="Открыть файл для дешифрования",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return

        content = read_file(file_path)
        if content is not None:
            self.decrypt_input_text.delete("1.0", tk.END)
            self.decrypt_input_text.insert("1.0", content)
        else:
            messagebox.showerror("Ошибка", "Не удалось прочитать файл. Проверьте формат и кодировку.")

    def save_file_encrypted(self):
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
                write_file(file_path, encrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_file_decrypted(self):
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
                write_file(file_path, decrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")


class RouteGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Маршрутный шифр")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.encrypt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.encrypt_frame, text="Шифрование")

        self.decrypt_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.decrypt_frame, text="Дешифрование")

        self.setup_encrypt_tab()
        self.setup_decrypt_tab()

    def setup_encrypt_tab(self):
        input_frame = ttk.LabelFrame(self.encrypt_frame, text="Исходный текст")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.encrypt_input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=60, height=10)
        self.encrypt_input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        load_btn = ttk.Button(input_frame, text="Загрузить из файла", command=self.open_file_for_encryption)
        load_btn.pack(anchor=tk.W, padx=5, pady=5)

        params_frame = ttk.LabelFrame(self.encrypt_frame, text="Параметры шифрования")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        width_frame = ttk.Frame(params_frame)
        width_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(width_frame, text="Ширина таблицы:").pack(side=tk.LEFT, padx=5)
        self.encrypt_width_var = tk.StringVar(value="11")
        width_entry = ttk.Entry(width_frame, textvariable=self.encrypt_width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)

        route_frame = ttk.Frame(params_frame)
        route_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(route_frame, text="Тип маршрута:").pack(side=tk.LEFT, padx=5)
        self.encrypt_route_var = tk.StringVar(value="спираль")
        route_combo = ttk.Combobox(route_frame, textvariable=self.encrypt_route_var, 
                                   values=["спираль", "змейка", "диагональ", "спираль_против", "зигзаг_верт"],
                                   state="readonly", width=15)
        route_combo.pack(side=tk.LEFT, padx=5)

        encrypt_btn = ttk.Button(self.encrypt_frame, text="Зашифровать", command=self.encrypt_text)
        encrypt_btn.pack(pady=10)

        output_frame = ttk.LabelFrame(self.encrypt_frame, text="Зашифрованный текст")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.encrypt_output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=10)
        self.encrypt_output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        save_btn = ttk.Button(output_frame, text="Сохранить в файл", command=self.save_file_encrypted)
        save_btn.pack(anchor=tk.E, padx=5, pady=5)

    def setup_decrypt_tab(self):
        input_frame = ttk.LabelFrame(self.decrypt_frame, text="Зашифрованный текст")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.decrypt_input_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=60, height=10)
        self.decrypt_input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        load_btn = ttk.Button(input_frame, text="Загрузить из файла", command=self.open_file_for_decryption)
        load_btn.pack(anchor=tk.W, padx=5, pady=5)

        params_frame = ttk.LabelFrame(self.decrypt_frame, text="Параметры дешифрования")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        width_frame = ttk.Frame(params_frame)
        width_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(width_frame, text="Ширина таблицы:").pack(side=tk.LEFT, padx=5)
        self.decrypt_width_var = tk.StringVar(value="11")
        width_entry = ttk.Entry(width_frame, textvariable=self.decrypt_width_var, width=5)
        width_entry.pack(side=tk.LEFT, padx=5)

        route_frame = ttk.Frame(params_frame)
        route_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(route_frame, text="Тип маршрута:").pack(side=tk.LEFT, padx=5)
        self.decrypt_route_var = tk.StringVar(value="")
        route_label = ttk.Label(route_frame, textvariable=self.decrypt_route_var, width=10)
        route_label.pack(side=tk.LEFT, padx=5)
        ttk.Label(route_frame, text="(определяется автоматически)").pack(side=tk.LEFT, padx=5)

        self.decrypt_height_var = tk.StringVar(value="")

        decrypt_btn = ttk.Button(self.decrypt_frame, text="Расшифровать", command=self.decrypt_text)
        decrypt_btn.pack(pady=10)

        output_frame = ttk.LabelFrame(self.decrypt_frame, text="Расшифрованный текст")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.decrypt_output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=60, height=10)
        self.decrypt_output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        save_frame = ttk.Frame(output_frame)
        save_frame.pack(fill=tk.X, padx=5, pady=5)

        save_btn = ttk.Button(save_frame, text="Сохранить текст", command=self.save_file_decrypted)
        save_btn.pack(side=tk.LEFT, padx=5)

    def open_file_for_encryption(self):
        file_path = filedialog.askopenfilename(
            title="Открыть файл для шифрования",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return

        content = read_file(file_path)
        if content is not None:
            self.encrypt_input_text.delete("1.0", tk.END)
            self.encrypt_input_text.insert("1.0", content)
        else:
            messagebox.showerror("Ошибка", "Не удалось прочитать файл. Проверьте формат и кодировку.")

    def encrypt_text(self):
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

            cipher = RouteCipher()
            encrypted, table = cipher.encrypt(text, width, route_type, remove_spaces=False)

            self.encrypt_output_text.delete("1.0", tk.END)
            self.encrypt_output_text.insert("1.0", encrypted)

        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при шифровании: {str(e)}")
            import traceback
            traceback.print_exc()

    def decrypt_text(self):
        text = self.decrypt_input_text.get("1.0", tk.END).strip()

        if not text:
            messagebox.showerror("Ошибка", "Введите текст для дешифрования")
            return

        try:
            width_str = self.decrypt_width_var.get().strip()
            if not width_str:
                messagebox.showerror("Ошибка", "Введите ширину таблицы для дешифрования")
                return

            width = int(width_str)
            height = (len(text) + width - 1) // width
            
            cipher = RouteCipher()
            
            route_type = cipher.analyze_route_pattern(text, width, height)
            self.decrypt_route_var.set(route_type)
            
            decrypted, table = cipher.decrypt(text, width, route_type=route_type)

            self.decrypt_output_text.delete("1.0", tk.END)
            self.decrypt_output_text.insert("1.0", decrypted)
            
            self.decrypt_height_var.set(str(height))
            
            messagebox.showinfo("Результат", f"Определен тип маршрута: {route_type}")

        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при дешифровании: {str(e)}")
            import traceback
            traceback.print_exc()

    def open_file_for_decryption(self):
        file_path = filedialog.askopenfilename(
            title="Открыть файл для дешифрования",
            filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return

        content = read_file(file_path)
        if content is not None:
            self.decrypt_input_text.delete("1.0", tk.END)
            self.decrypt_input_text.insert("1.0", content)
        else:
            messagebox.showerror("Ошибка", "Не удалось прочитать файл. Проверьте формат и кодировку.")

    def save_file_encrypted(self):
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
                write_file(file_path, encrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_file_decrypted(self):
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
                write_file(file_path, decrypted_text)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def save_with_info(self, tab_index):
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
            return

        try:
            encrypted_text = self.decrypt_input_text.get("1.0", tk.END).strip()

            width = self.decrypt_width_var.get().strip()
            height = self.decrypt_height_var.get().strip()
            route_type = self.decrypt_route_var.get().strip()

            full_content = f"=== ЗАШИФРОВАННЫЙ ТЕКСТ ===\n{encrypted_text}\n\n"
            full_content += f"=== ИНФОРМАЦИЯ О РАСШИФРОВКЕ ===\n"
            full_content += f"Размер таблицы: {width}x{height}\n"
            full_content += f"Тип маршрута: {route_type}\n\n"
            full_content += f"=== РАСШИФРОВАННЫЙ ТЕКСТ ===\n{content}"

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
