"""
Модуль genLabelSet (Генератор набора меток для обучения)
Создает одно-hot закодированные метки для классификации на 3 класса.

Назначение:
- Генерирует метки для 2100 примеров
- 3 класса: 0, 1, 2
- Каждый класс содержит по 700 примеров
- Сохраняет результат в файл label_set.txt

Формат выходного файла:
- Каждая строка - один пример
- Три числа (0 или 1), разделенные запятыми
- Пример: "1,0,0" означает класс 0
          "0,1,0" означает класс 1
          "0,0,1" означает класс 2
"""

import numpy as np
import matplotlib.pyplot as plt


def save_file(data_list, filename):
    """
    Сохраняет двумерный массив в CSV файл.
    
    Аргументы:
        data_list (list или numpy.ndarray): Двумерный массив данных
        filename (str): Имя файла для сохранения
        
    Пример:
        save_file([[1,0,0], [0,1,0]], "labels.txt")
        # Создаст файл с содержимым:
        # 1,0,0
        # 0,1,0
    """
    with open(filename, "w") as file:
        for row in data_list:
            # Проходим по каждому элементу строки
            for i in range(len(row)):
                # Добавляем запятую после каждого элемента, кроме последнего
                if i != len(row) - 1:
                    file.write(str(row[i]) + ",")
                else:
                    file.write(str(row[i]))
            # Переход на новую строку после каждого примера
            file.write("\n")
    
    print(f"Файл '{filename}' успешно сохранен")
    print(f"Размер: {len(data_list)} строк, {len(data_list[0]) if len(data_list) > 0 else 0} колонок")


def generate_one_hot_labels(num_samples_per_class=700, num_classes=3):
    """
    Генерирует одно-hot закодированные метки для классификации.
    
    Аргументы:
        num_samples_per_class (int): Количество примеров на каждый класс
        num_classes (int): Количество классов (по умолчанию 3)
        
    Возвращает:
        numpy.ndarray: Массив одно-hot меток размером (total_samples, num_classes)
        
    Пример:
        labels = generate_one_hot_labels(2, 3)
        # Результат: [[1,0,0], [1,0,0], [0,1,0], [0,1,0], [0,0,1], [0,0,1]]
    """
    total_samples = num_samples_per_class * num_classes
    
    # Создаем массив с номерами классов
    # [0,0,0,..., 1,1,1,..., 2,2,2,...]
    labels = np.array([i for i in range(num_classes) for _ in range(num_samples_per_class)])
    
    # Создаем пустой массив для одно-hot меток
    one_hot_labels = np.zeros((total_samples, num_classes))
    
    # Заполняем одно-hot векторы
    for i in range(total_samples):
        one_hot_labels[i, labels[i]] = 1
    
    return one_hot_labels, labels


def display_statistics(one_hot_labels, labels):
    """
    Выводит статистику о сгенерированных метках.
    
    Аргументы:
        one_hot_labels (numpy.ndarray): Одно-hot метки
        labels (numpy.ndarray): Исходные метки классов
    """
    print("\n" + "="*50)
    print("СТАТИСТИКА СГЕНЕРИРОВАННЫХ МЕТОК")
    print("="*50)
    
    print(f"\nОбщее количество примеров: {len(labels)}")
    print(f"Количество классов: {one_hot_labels.shape[1]}")
    
    print("\nРаспределение по классам:")
    for class_num in range(one_hot_labels.shape[1]):
        count = np.sum(labels == class_num)
        print(f"  Класс {class_num}: {count} примеров")
    
    print(f"\nФорма одно-hot массива: {one_hot_labels.shape}")
    
    print("\nПервые 5 примеров одно-hot меток:")
    for i in range(min(5, len(one_hot_labels))):
        print(f"  Пример {i+1}: {one_hot_labels[i]} -> класс {np.argmax(one_hot_labels[i])}")
    
    print("\nПоследние 5 примеров одно-hot меток:")
    for i in range(max(0, len(one_hot_labels)-5), len(one_hot_labels)):
        print(f"  Пример {i+1}: {one_hot_labels[i]} -> класс {np.argmax(one_hot_labels[i])}")
    
    print("="*50 + "\n")


def visualize_distribution(labels, num_classes=3):
    """
    Визуализирует распределение классов в виде гистограммы.
    
    Аргументы:
        labels (numpy.ndarray): Массив с номерами классов
        num_classes (int): Количество классов
    """
    plt.figure(figsize=(8, 5))
    
    # Подсчитываем количество примеров в каждом классе
    class_counts = [np.sum(labels == i) for i in range(num_classes)]
    
    # Создаем столбчатую диаграмму
    plt.bar(range(num_classes), class_counts, color=['red', 'green', 'blue'])
    
    # Настройка графика
    plt.xlabel('Класс')
    plt.ylabel('Количество примеров')
    plt.title('Распределение меток по классам')
    plt.xticks(range(num_classes), [f'Класс {i}' for i in range(num_classes)])
    
    # Добавляем значения на столбцы
    for i, count in enumerate(class_counts):
        plt.text(i, count + 10, str(count), ha='center', va='bottom')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('labels_distribution.png', dpi=150)
    print("График распределения сохранен в 'labels_distribution.png'")
    
    # Показываем график
    plt.show()


# ==================== ОСНОВНАЯ ГЕНЕРАЦИЯ ====================

if __name__ == "__main__":
    """
    Основной блок генерации меток.
    Выполняется только при запуске скрипта напрямую.
    """
    
    # Параметры генерации (можно легко изменить)
    SAMPLES_PER_CLASS = 700   # Количество примеров на класс
    NUM_CLASSES = 3           # Количество классов
    
    print("ГЕНЕРАТОР ОДНО-HOT МЕТОК")
    print("="*50)
    print(f"Классов: {NUM_CLASSES}")
    print(f"Примеров на класс: {SAMPLES_PER_CLASS}")
    print(f"Всего примеров: {SAMPLES_PER_CLASS * NUM_CLASSES}")
    print("="*50)
    
    # Генерируем метки
    one_hot_labels, labels = generate_one_hot_labels(SAMPLES_PER_CLASS, NUM_CLASSES)
    
    # Выводим статистику
    display_statistics(one_hot_labels, labels)
    
    # Сохраняем в файл
    save_file(one_hot_labels, "label_set.txt")
    
    # Визуализируем распределение (опционально)
    visualize_distribution(labels, NUM_CLASSES)
    
    # Дополнительная информация о формате файла
    print("\nФайл 'label_set.txt' готов к использованию в нейросети.")
    print("Каждая строка содержит одно-hot вектор:")
    print("  - Класс 0: [1, 0, 0]")
    print("  - Класс 1: [0, 1, 0]")
    print("  - Класс 2: [0, 0, 1]")