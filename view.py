"""
Модуль View (Представление) для приложения NeuroCanvas
Отвечает за создание и отображение пользовательского интерфейса

Модуль содержит класс View, который управляет всеми визуальными элементами:
- Левая панель: настройка архитектуры нейросети, параметры обучения
- Центральная панель: Canvas для визуализации нейросети
- Правая панель: отображение ошибок и результатов предсказаний
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import config as cfg

# Импорт визуальных компонентов
from node import Node
from arrow import Arrow
from value_box import ValueBox


class View:
    """ 
    Класс View отвечает за создание и управление графическим интерфейсом.
    Реализует паттерн MVC (Model-View-Controller) - представление.
    
    Атрибуты:
        container: родительское окно (главное окно приложения)
        controller: ссылка на контроллер (нейросеть)
        leftFrame: левая панель с настройками
        midFrame: центральная панель с Canvas для рисования
        rightFrame: правая панель с результатами
        c: Canvas для отрисовки нейросети
        statusbar: строка состояния внизу
        listInputNode: список входных узлов
        listOutputNode: список выходных узлов
        ... и множество виджетов для управления нейросетью
    """
    
    def __init__(self, parent):
        """ 
        Конструктор класса View.
        
        Аргументы:
            parent: родительский элемент (главное окно tkinter.Tk())
        """
        # Сохраняем ссылку на родительское окно (точка входа в GUI)
        self.container = parent
        self.controller = None
        
        # Списки для хранения визуальных элементов
        self.listInputNode = []
        self.listOutputNode = []
        self.listHiddenNodes = []
        self.listArrows = []
        self.listValueBox = []
    
    def setController(self, controller):
        """
        Устанавливает связь с контроллером.
        
        Аргументы:
            controller: экземпляр класса NeuralNetwork
        """
        self.controller = controller
    
    def setup(self):
        """ 
        Главный метод настройки пользовательского интерфейса.
        Вызывает методы создания виджетов и их размещения.
        """
        self.create_widgets()    # Создаем все виджеты
        self.setup_layout()      # Размещаем их на окне
        
        # Привязываем события к кнопкам
        self.bind_events()
    
    def bind_events(self):
        """
        Привязывает события к кнопкам интерфейса.
        """
        # Кнопка создания нейросети
        self.b2Circle.config(command=self.create_network)
        
        # Кнопки обучения
        self.b8LoadFeature.config(command=self.load_train_features)
        self.b9LoadLabel.config(command=self.load_train_labels)
        self.b10StartTrain.config(command=self.start_training)
        
        # Кнопки сохранения/загрузки
        self.b10aSaveNN.config(command=self.save_network)
        self.b10bLoadNN.config(command=self.load_network)
        
        # Кнопки прогнозирования
        self.b11LoadPreFeature.config(command=self.load_test_features)
        self.b12StartPredict.config(command=self.start_prediction)
    
    def create_widgets(self):
        """ 
        Метод создания всех виджетов интерфейса.
        Создает рамки, кнопки, поля ввода, Canvas и другие элементы.
        """
        
        # ==================== ОСНОВНЫЕ РАМКИ (ФРЕЙМЫ) ====================
        # Левая рамка - для настроек архитектуры и обучения
        self.leftFrame = tk.Frame(self.container, width=100, height=cfg.windowHeight)
        
        # Центральная рамка - для отрисовки нейросети
        self.midFrame = tk.Frame(self.container, width=cfg.canvasWidth, height=cfg.canvasHeight)
        
        # Правая рамка - для вывода результатов
        self.rightFrame = tk.Frame(self.container, width=100, height=cfg.windowHeight)
        
        # Внутренние рамки внутри правой (для организации пространства)
        self.rightFrame1 = tk.Frame(self.rightFrame)  # Верхняя часть (ошибки)
        self.rightFrame2 = tk.Frame(self.rightFrame)  # Нижняя часть (предсказания)
        
        # ==================== СТРОКА СОСТОЯНИЯ ====================
        # Отображает текущий статус работы программы
        self.statusbar = tk.Label(
            self.midFrame,
            text="Готов.",              # Начальный текст
            bd=1,                       # Граница (border)
            relief=tk.SUNKEN,           # Утопленный стиль
            anchor=tk.W,               # Выравнивание по левому краю
            fg="red"                   # Красный цвет текста
        )
        
        # ==================== РАЗДЕЛ: АРХИТЕКТУРА НЕЙРОСЕТИ ====================
        # Виджеты для настройки количества нейронов в каждом слое
        
        # Входной слой (input layer)
        self.lab1input = tk.Label(self.leftFrame, text="Входной слой")
        self.combo1input = ttk.Combobox(self.leftFrame, values=[1, 2, 3, 4])
        
        # Скрытый слой 1 (hidden layer 1)
        self.lab2layer1 = tk.Label(self.leftFrame, text="Скрытый слой 1")
        self.combo2layer1 = ttk.Combobox(
            self.leftFrame, 
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        
        # Скрытый слой 2 (hidden layer 2)
        self.lab3layer2 = tk.Label(self.leftFrame, text="Скрытый слой 2")
        self.combo3layer2 = ttk.Combobox(
            self.leftFrame, 
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 = слой отсутствует
        )
        
        # Скрытый слой 3 (hidden layer 3)
        self.lab4layer3 = tk.Label(self.leftFrame, text="Скрытый слой 3")
        self.combo4layer3 = ttk.Combobox(
            self.leftFrame, 
            values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0 = слой отсутствует
        )
        
        # Выходной слой (output layer)
        self.lab5output = tk.Label(self.leftFrame, text="Выходной слой")
        self.combo5output = ttk.Combobox(
            self.leftFrame, 
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
        
        # Кнопка создания нейросети на основе выбранной архитектуры
        self.b2Circle = tk.Button(
            self.leftFrame,
            text="Создать нейросеть",
            width=20,
            height=1
        )
        
        # ==================== РАЗДЕЛ: ОБУЧЕНИЕ НЕЙРОСЕТИ ====================
        self.lab6TrainDiv = tk.Label(self.leftFrame, text="=== Обучение ===")
        
        # Скорость обучения (learning rate)
        self.lab7LearningRate = tk.Label(self.leftFrame, text="Скорость обучения")
        self.combo6learningRate = ttk.Combobox(
            self.leftFrame,
            values=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        # Количество эпох (epochs)
        self.lab8epoch = tk.Label(self.leftFrame, text="Количество эпох")
        self.combo7epoch = ttk.Combobox(
            self.leftFrame,
            values=[
                10, 5000, 10000, 50000, 100000, 200000, 500000, 1000000,
                2000000, 5000000, 10000000, 20000000, 50000000
            ]
        )
        
        # Частота обновления экрана (через сколько эпох перерисовывать)
        self.lab9refreshRate = tk.Label(self.leftFrame, text="Частота обновления (эпох)")
        self.combo8refreshRate = ttk.Combobox(
            self.leftFrame,
            values=[2, 100, 200, 300, 500, 1000, 2000, 3000]
        )
        
        # Кнопки для обучения
        self.b8LoadFeature = tk.Button(
            self.leftFrame,
            text="Загрузить хар-ки НС",
            width=20,
            height=1
        )
        self.b9LoadLabel = tk.Button(
            self.leftFrame,
            text="Загрузить метки НС",
            width=20,
            height=1
        )
        self.b10StartTrain = tk.Button(
            self.leftFrame,
            text="Начать обучение",
            width=20,
            height=1
        )
        
        # ==================== РАЗДЕЛ: СОХРАНЕНИЕ / ЗАГРУЗКА ====================
        self.lab9aSaveLoadFile = tk.Label(self.leftFrame, text="=== Сохранить / Загрузить ===")
        self.b10aSaveNN = tk.Button(
            self.leftFrame,
            text="Сохранить нейросеть",
            width=20,
            height=1
        )
        self.b10bLoadNN = tk.Button(
            self.leftFrame,
            text="Загрузить нейросеть",
            width=20,
            height=1
        )
        
        # ==================== РАЗДЕЛ: ПРОГНОЗИРОВАНИЕ ====================
        self.lab10DivPredection = tk.Label(self.leftFrame, text="=== Прогнозирование ===")
        
        self.b11LoadPreFeature = tk.Button(
            self.leftFrame,
            text="Загрузить ф-цию прогноз.",
            width=20,
            height=1
        )
        self.b12StartPredict = tk.Button(
            self.leftFrame,
            text="Начать прогнозирование",
            width=20,
            height=1
        )
        
        # Округление результатов предсказания
        self.lab11PreRounding = tk.Label(self.leftFrame, text="=== Округление ===")
        self.combo9PreRounding = ttk.Combobox(
            self.leftFrame,
            values=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        )
        
        # ==================== ЦЕНТРАЛЬНАЯ ОБЛАСТЬ (CANVAS) ====================
        # Холст для отрисовки структуры нейросети
        self.c = tk.Canvas(
            self.midFrame,
            bg='white',
            width=cfg.canvasWidth,
            height=cfg.canvasHeight
        )
        
        # ==================== ПРАВАЯ ОБЛАСТЬ (ВЫВОД РЕЗУЛЬТАТОВ) ====================
        
        # Блок "Функция потерь нейросети" (ошибки)
        self.lab31ErrorRate = tk.Label(self.rightFrame, text="=== Функция потерь НС ===")
        
        # Текстовое поле для вывода ошибок с полосой прокрутки
        self.text31ErrorTxtbox = tk.Text(self.rightFrame1, width=30)
        self.scl31vbar = tk.Scrollbar(self.rightFrame1, orient=tk.VERTICAL)
        self.text31ErrorTxtbox.config(yscrollcommand=self.scl31vbar.set)
        self.scl31vbar.config(command=self.text31ErrorTxtbox.yview)
        
        # Блок "Результат предсказания"
        self.lab32PredictionResult = tk.Label(self.rightFrame, text="=== Результат предсказания ===")
        
        # Текстовое поле для вывода предсказаний с полосой прокрутки
        self.text32PredictTxtbox = tk.Text(self.rightFrame2, width=30)
        self.scl32vbarPre = tk.Scrollbar(self.rightFrame2, orient=tk.VERTICAL)
        self.text32PredictTxtbox.config(yscrollcommand=self.scl32vbarPre.set)
        self.scl32vbarPre.config(command=self.text32PredictTxtbox.yview)
    
    def setup_layout(self):
        """ 
        Метод размещения виджетов на экране.
        Использует менеджер геометрии pack() для организации интерфейса.
        """
        
        # ==================== РАЗМЕЩЕНИЕ ОСНОВНЫХ РАМОК ====================
        self.leftFrame.pack(side=tk.LEFT, fill=tk.Y)    # Левая - по левому краю
        self.midFrame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)  # Центр - расширяется
        self.rightFrame.pack(side=tk.RIGHT, fill=tk.Y)  # Правая - по правому краю
        
        # ==================== ЛЕВАЯ ПАНЕЛЬ: АРХИТЕКТУРА ====================
        self.lab1input.pack(side=tk.TOP, pady=2)
        self.combo1input.pack(side=tk.TOP, pady=2)
        self.lab2layer1.pack(side=tk.TOP, pady=2)
        self.combo2layer1.pack(side=tk.TOP, pady=2)
        self.lab3layer2.pack(side=tk.TOP, pady=2)
        self.combo3layer2.pack(side=tk.TOP, pady=2)
        self.lab4layer3.pack(side=tk.TOP, pady=2)
        self.combo4layer3.pack(side=tk.TOP, pady=2)
        self.lab5output.pack(side=tk.TOP, pady=2)
        self.combo5output.pack(side=tk.TOP, pady=2)
        self.b2Circle.pack(side=tk.TOP, pady=10)
        
        # ==================== ЛЕВАЯ ПАНЕЛЬ: ОБУЧЕНИЕ ====================
        self.lab6TrainDiv.pack(side=tk.TOP, pady=(10, 5))
        self.lab7LearningRate.pack(side=tk.TOP, pady=2)
        self.combo6learningRate.pack(side=tk.TOP, pady=2)
        self.lab8epoch.pack(side=tk.TOP, pady=2)
        self.combo7epoch.pack(side=tk.TOP, pady=2)
        self.lab9refreshRate.pack(side=tk.TOP, pady=2)
        self.combo8refreshRate.pack(side=tk.TOP, pady=2)
        self.b8LoadFeature.pack(side=tk.TOP, pady=5)
        self.b9LoadLabel.pack(side=tk.TOP, pady=5)
        self.b10StartTrain.pack(side=tk.TOP, pady=5)
        
        # ==================== ЛЕВАЯ ПАНЕЛЬ: СОХРАНЕНИЕ/ЗАГРУЗКА ====================
        self.lab9aSaveLoadFile.pack(side=tk.TOP, pady=(10, 5))
        self.b10aSaveNN.pack(side=tk.TOP, pady=5)
        self.b10bLoadNN.pack(side=tk.TOP, pady=5)
        
        # ==================== ЛЕВАЯ ПАНЕЛЬ: ПРОГНОЗИРОВАНИЕ ====================
        self.lab10DivPredection.pack(side=tk.TOP, pady=(10, 5))
        self.b11LoadPreFeature.pack(side=tk.TOP, pady=5)
        self.b12StartPredict.pack(side=tk.TOP, pady=5)
        self.lab11PreRounding.pack(side=tk.TOP, pady=2)
        self.combo9PreRounding.pack(side=tk.TOP, pady=2)
        
        # ==================== ПРАВАЯ ПАНЕЛЬ: ОШИБКИ ====================
        self.lab31ErrorRate.pack(side=tk.TOP, pady=5)
        self.rightFrame1.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        self.text31ErrorTxtbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scl31vbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ==================== ПРАВАЯ ПАНЕЛЬ: ПРЕДСКАЗАНИЯ ====================
        self.lab32PredictionResult.pack(side=tk.TOP, pady=5)
        self.rightFrame2.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=5)
        self.text32PredictTxtbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scl32vbarPre.pack(side=tk.RIGHT, fill=tk.Y)
        
        # ==================== ЦЕНТРАЛЬНАЯ ПАНЕЛЬ (CANVAS) ====================
        self.c.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        
        # ==================== СТРОКА СОСТОЯНИЯ ====================
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # ==================== УСТАНОВКА ЗНАЧЕНИЙ ПО УМОЛЧАНИЮ ====================
        
        # Архитектура нейросети по умолчанию
        self.combo1input.current(1)      # Входной слой: 2 нейрона
        self.combo2layer1.current(3)     # Скрытый слой 1: 4 нейрона
        self.combo3layer2.current(0)     # Скрытый слой 2: отключен (0)
        self.combo4layer3.current(0)     # Скрытый слой 3: отключен (0)
        self.combo5output.current(2)     # Выходной слой: 3 нейрона
        
        # Параметры обучения по умолчанию
        self.combo6learningRate.current(2)  # Скорость обучения: 0.001
        self.combo7epoch.current(1)         # Количество эпох: 5000
        self.combo8refreshRate.current(2)   # Частота обновления: 200 эпох
        
        # Округление по умолчанию
        self.combo9PreRounding.current(1)   # 1 (без округления)
        
        # Финальное сообщение в статусной строке
        self.statusbar.config(text="Готов. Выберите параметры нейросети и нажмите 'Создать нейросеть'")
    
    # ==================== МЕТОДЫ-ЗАГЛУШКИ (ДОЛЖНЫ БЫТЬ РЕАЛИЗОВАНЫ) ====================
    
    def create_network(self):
        """Создает нейросеть на основе выбранных параметров."""
        if self.controller:
            self.controller.create_network()
    
    def load_train_features(self):
        """Загружает обучающие признаки."""
        if self.controller:
            filename = filedialog.askopenfilename(title="Выберите файл с признаками")
            self.controller.loadTrainFeature(filename)
    
    def load_train_labels(self):
        """Загружает обучающие метки."""
        if self.controller:
            filename = filedialog.askopenfilename(title="Выберите файл с метками")
            self.controller.loadTrainLabel(filename)
    
    def start_training(self):
        """Запускает обучение нейросети."""
        if self.controller:
            self.controller.startTrain()
    
    def save_network(self):
        """Сохраняет нейросеть в файл."""
        if self.controller:
            filename = filedialog.asksaveasfilename(title="Сохранить нейросеть", defaultextension=".npy")
            self.controller.saveNetwork(filename)
    
    def load_network(self):
        """Загружает нейросеть из файла."""
        if self.controller:
            filename = filedialog.askopenfilename(title="Загрузить нейросеть")
            self.controller.loadNetwork(filename)
    
    def load_test_features(self):
        """Загружает тестовые признаки для прогнозирования."""
        if self.controller:
            filename = filedialog.askopenfilename(title="Выберите файл с тестовыми признаками")
            self.controller.loadTestFeature(filename)
    
    def start_prediction(self):
        """Запускает прогнозирование."""
        if self.controller:
            self.controller.startPredict()
    
    def updateWeight(self, weights):
        """Обновляет отображение весов."""
        pass  # Будет реализовано
    
    def updateBias(self, biases):
        """Обновляет отображение смещений."""
        pass  # Будет реализовано
    
    def clearInputValue(self):
        """Очищает значения входных узлов."""
        pass  # Будет реализовано
    
    def clearOutputValue(self):
        """Очищает значения выходных узлов."""
        pass  # Будет реализовано


# ==================== ТЕСТОВЫЙ ЗАПУСК (ДЛЯ ОТЛАДКИ) ====================
if __name__ == "__main__":
    """
    Этот блок выполняется только при запуске файла напрямую.
    Позволяет протестировать интерфейс независимо от основного приложения.
    """
    mainwin = tk.Tk()
    WIDTH = cfg.windowWidth
    HEIGHT = cfg.windowHeight
    mainwin.geometry(f"{WIDTH}x{HEIGHT}")
    mainwin.title("Тестовое окно - View")
    
    # Создаем экземпляр View и настраиваем интерфейс
    view = View(mainwin)
    view.setup()
    
    # Запускаем главный цикл
    mainwin.mainloop()