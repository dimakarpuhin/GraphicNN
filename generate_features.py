"""
Модуль genFeatureSet (Генератор набора признаков для обучения)
Создает синтетические двумерные данные для трех классов с гауссовским распределением.

Назначение:
- Генерирует признаки для 3 классов объектов
- Каждый класс имеет свое центроидное расположение в 2D пространстве
- Всего 2100 примеров (по 700 на класс)
- Сохраняет результат в файл feature_gen_set.txt

Классы и их центры:
- Кошки (класс 0): центр в точке (0, -3)
- Мыши (класс 1): центр в точке (3, 3)
- Собаки (класс 2): центр в точке (-3, 3)

Формат выходного файла:
- Каждая строка - один пример (две координаты)
- Координаты разделены запятыми
- Пример: "0.123, -2.890" (кошка)
          "2.456, 2.789" (мышь)
          "-3.123, 2.456" (собака)
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
        save_file([[1.2, 3.4], [5.6, 7.8]], "data.txt")
        # Создаст файл с содержимым:
        # 1.2,3.4
        # 5.6,7.8
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


def generate_class_data(center_x, center_y, num_samples=700, std_dev=1.0):
    """
    Генерирует данные для одного класса с гауссовским распределением.
    
    Аргументы:
        center_x (float): X-координата центра класса
        center_y (float): Y-координата центра класса
        num_samples (int): Количество примеров в классе
        std_dev (float): Стандартное отклонение (разброс данных)
        
    Возвращает:
        numpy.ndarray: Массив признаков размером (num_samples, 2)
        
    Пример:
        cat_data = generate_class_data(0, -3, 700, 1.0)
    """
    # Генерируем точки с нормальным распределением вокруг центра
    data = np.random.randn(num_samples, 2) * std_dev + np.array([center_x, center_y])
    return data


def generate_feature_set(class_configs, samples_per_class=700, random_seed=41):
    """
    Генерирует полный набор признаков для всех классов.
    
    Аргументы:
        class_configs (list): Список конфигураций классов
            Каждая конфигурация: (имя_класса, x_center, y_center, цвет)
        samples_per_class (int): Количество примеров на класс
        random_seed (int): Seed для воспроизводимости
        
    Возвращает:
        tuple: (feature_set, labels, class_names)
            feature_set: массив признаков
            labels: массив меток классов
            class_names: список имен классов
    """
    # Устанавливаем seed для воспроизводимости
    np.random.seed(random_seed)
    
    feature_sets = []
    labels = []
    
    print("Генерация данных для классов:")
    print("-" * 40)
    
    for class_idx, (class_name, center_x, center_y, color) in enumerate(class_configs):
        # Генерируем данные для текущего класса
        class_data = generate_class_data(center_x, center_y, samples_per_class)
        feature_sets.append(class_data)
        
        # Создаем метки для текущего класса
        class_labels = np.full(samples_per_class, class_idx)
        labels.append(class_labels)
        
        print(f"  {class_name}: {samples_per_class} примеров, центр=({center_x}, {center_y})")
    
    # Объединяем данные всех классов
    feature_set = np.vstack(feature_sets)
    all_labels = np.hstack(labels)
    
    return feature_set, all_labels, [name for name, _, _, _ in class_configs]


def display_feature_statistics(feature_set, labels, class_names):
    """
    Выводит статистику о сгенерированных признаках.
    
    Аргументы:
        feature_set (numpy.ndarray): Массив признаков
        labels (numpy.ndarray): Массив меток классов
        class_names (list): Список имен классов
    """
    print("\n" + "="*60)
    print("СТАТИСТИКА СГЕНЕРИРОВАННЫХ ПРИЗНАКОВ")
    print("="*60)
    
    print(f"\nОбщее количество примеров: {len(feature_set)}")
    print(f"Количество признаков: {feature_set.shape[1]}")
    print(f"Количество классов: {len(class_names)}")
    
    print("\nСтатистика по каждому классу:")
    for i, class_name in enumerate(class_names):
        class_data = feature_set[labels == i]
        print(f"\n  {class_name}:")
        print(f"    Количество: {len(class_data)}")
        print(f"    X: min={class_data[:, 0].min():.3f}, max={class_data[:, 0].max():.3f}, "
              f"mean={class_data[:, 0].mean():.3f}, std={class_data[:, 0].std():.3f}")
        print(f"    Y: min={class_data[:, 1].min():.3f}, max={class_data[:, 1].max():.3f}, "
              f"mean={class_data[:, 1].mean():.3f}, std={class_data[:, 1].std():.3f}")
    
    print("\n" + "="*60)


def visualize_feature_set(feature_set, labels, class_names, save_image=True):
    """
    Визуализирует набор признаков на 2D плоскости.
    
    Аргументы:
        feature_set (numpy.ndarray): Массив признаков (N x 2)
        labels (numpy.ndarray): Массив меток классов
        class_names (list): Список имен классов
        save_image (bool): Сохранять ли изображение
    """
    # Цвета для каждого класса
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    plt.figure(figsize=(10, 8))
    
    # Отображаем точки каждого класса
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        plt.scatter(
            feature_set[class_mask, 0],
            feature_set[class_mask, 1],
            c=colors[i % len(colors)],
            label=class_name,
            alpha=0.6,
            s=30
        )
    
    # Настройка графика
    plt.xlabel('Признак X', fontsize=12)
    plt.ylabel('Признак Y', fontsize=12)
    plt.title('Визуализация набора признаков для нейросети', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Добавляем центры классов
    class_centers = []
    for i in range(len(class_names)):
        class_data = feature_set[labels == i]
        center_x, center_y = class_data.mean(axis=0)
        class_centers.append((center_x, center_y))
        plt.plot(center_x, center_y, 'o', color=colors[i % len(colors)], 
                markersize=10, markeredgecolor='black', markeredgewidth=2)
        plt.annotate(f'{class_names[i]}', (center_x, center_y), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    if save_image:
        plt.savefig('feature_set_visualization.png', dpi=150, bbox_inches='tight')
        print("\nГрафик сохранен в 'feature_set_visualization.png'")
    
    plt.show()


def split_train_test(feature_set, labels, train_ratio=0.8, random_seed=42):
    """
    Разделяет данные на обучающую и тестовую выборки.
    
    Аргументы:
        feature_set (numpy.ndarray): Массив признаков
        labels (numpy.ndarray): Массив меток
        train_ratio (float): Доля обучающих данных (0-1)
        random_seed (int): Seed для воспроизводимости
        
    Возвращает:
        tuple: (train_features, test_features, train_labels, test_labels)
    """
    np.random.seed(random_seed)
    
    # Создаем индексы и перемешиваем
    indices = np.arange(len(feature_set))
    np.random.shuffle(indices)
    
    # Разделяем индексы
    split_point = int(len(feature_set) * train_ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    # Разделяем данные
    train_features = feature_set[train_indices]
    test_features = feature_set[test_indices]
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    
    print(f"\nРазделение данных:")
    print(f"  Обучающая выборка: {len(train_features)} примеров ({train_ratio*100:.0f}%)")
    print(f"  Тестовая выборка: {len(test_features)} примеров ({(1-train_ratio)*100:.0f}%)")
    
    return train_features, test_features, train_labels, test_labels


def save_class_datasets(train_features, test_features, train_labels, test_labels):
    """
    Сохраняет разделенные наборы данных в отдельные файлы.
    
    Аргументы:
        train_features: обучающие признаки
        test_features: тестовые признаки
        train_labels: обучающие метки
        test_labels: тестовые метки
    """
    # Сохраняем обучающие признаки
    save_file(train_features, "train_features.txt")
    
    # Сохраняем тестовые признаки
    save_file(test_features, "test_features.txt")
    
    # Сохраняем обучающие метки (в формате для нейросети)
    # Преобразуем в одно-hot кодировку
    train_one_hot = np.zeros((len(train_labels), len(np.unique(train_labels))))
    for i, label in enumerate(train_labels):
        train_one_hot[i, int(label)] = 1
    save_file(train_one_hot, "train_labels.txt")
    
    # Сохраняем тестовые метки (в формате для нейросети)
    test_one_hot = np.zeros((len(test_labels), len(np.unique(test_labels))))
    for i, label in enumerate(test_labels):
        test_one_hot[i, int(label)] = 1
    save_file(test_one_hot, "test_labels.txt")


# ==================== ОСНОВНАЯ ГЕНЕРАЦИЯ ====================

if __name__ == "__main__":
    """
    Основной блок генерации признаков.
    Выполняется только при запуске скрипта напрямую.
    """
    
    # Параметры генерации
    SAMPLES_PER_CLASS = 700   # Количество примеров на класс
    RANDOM_SEED = 41          # Seed для воспроизводимости
    
    # Конфигурация классов: (имя, x_центр, y_центр, цвет)
    CLASS_CONFIGS = [
        ("Кошки (класс 0)", 0, -3, "red"),
        ("Мыши (класс 1)", 3, 3, "green"),
        ("Собаки (класс 2)", -3, 3, "blue"),
    ]
    
    print("ГЕНЕРАТОР НАБОРА ПРИЗНАКОВ")
    print("="*60)
    print(f"Классов: {len(CLASS_CONFIGS)}")
    print(f"Примеров на класс: {SAMPLES_PER_CLASS}")
    print(f"Всего примеров: {SAMPLES_PER_CLASS * len(CLASS_CONFIGS)}")
    print("="*60 + "\n")
    
    # Генерируем признаки
    feature_set, labels, class_names = generate_feature_set(
        CLASS_CONFIGS, 
        SAMPLES_PER_CLASS, 
        RANDOM_SEED
    )
    
    # Выводим статистику
    display_feature_statistics(feature_set, labels, class_names)
    
    # Сохраняем полный набор признаков
    save_file(feature_set, "feature_gen_set.txt")
    
    # Разделяем на обучающую и тестовую выборки
    train_features, test_features, train_labels, test_labels = split_train_test(
        feature_set, labels, train_ratio=0.8
    )
    
    # Сохраняем разделенные наборы
    save_class_datasets(train_features, test_features, train_labels, test_labels)
    
    # Визуализируем данные
    visualize_feature_set(feature_set, labels, class_names, save_image=True)
    
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("="*60)
    print("\nСозданные файлы:")
    print("  1. feature_gen_set.txt     - полный набор признаков (2100 примеров)")
    print("  2. train_features.txt      - обучающие признаки (1680 примеров)")
    print("  3. test_features.txt       - тестовые признаки (420 примеров)")
    print("  4. train_labels.txt        - обучающие метки (одно-hot формат)")
    print("  5. test_labels.txt         - тестовые метки (одно-hot формат)")
    print("  6. feature_set_visualization.png - визуализация данных")
    print("\nЭти файлы можно использовать для обучения нейросети в вашем проекте!")