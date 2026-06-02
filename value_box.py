"""
Модуль value_box (Визуальное отображение значений)
Содержит класс ValueBox, который создает и управляет прямоугольными блоками
для отображения числовых значений (весов, смещений) на холсте.

Каждый объект ValueBox представляет собой:
- Прямоугольник с рамкой
- Текстовое поле с числовым значением
- Возможность связывания с левым и правым узлами (нейронами)
"""

import config as cfg
import random


class ValueBox:
    """
    Класс для визуализации числовых значений (весов или смещений) на Canvas.
    
    Атрибуты:
        value (float): числовое значение для отображения
        rightNode: ссылка на правый узел (нейрон, к которому ведет значение)
        leftNode: ссылка на левый узел (нейрон, от которого идет значение)
        what (str): тип объекта ("valuebox")
        canvas: ссылка на холст tkinter.Canvas
        centreX, centreY: координаты центра блока
        rid: ID прямоугольника на Canvas
        tId: ID текстового поля на Canvas
    """
    
    def __init__(self):
        """
        Конструктор класса ValueBox.
        Инициализирует пустой объект с нулевым значением.
        """
        self.value = 0.00                      # Отображаемое числовое значение
        self.rightNode = None                  # Правый узел (нейрон)
        self.leftNode = None                   # Левый узел (нейрон)
        self.what = "valuebox"                 # Идентификатор типа объекта
        
        # Эти атрибуты будут инициализированы позже в методах create() и updatePoints()
        self.canvas = None
        self.centreX = 0
        self.centreY = 0
        self.startX = self.startY = 0
        self.endX = self.endY = 0
        self.leftX = self.leftY = 0
        self.rightX = self.rightY = 0
        self.textX = self.textY = 0
        self.rid = None
        self.tId = None
    
    def updatePoints(self):
        """
        Пересчитывает координаты всех элементов блока на основе текущего центра.
        Вызывается при создании объекта и при необходимости обновления позиции.
        
        Вычисляет:
            - startX, startY: левый верхний угол прямоугольника
            - endX, endY: правый нижний угол прямоугольника
            - leftX, leftY: центральная точка левой стороны
            - rightX, rightY: центральная точка правой стороны
            - textX, textY: центр прямоугольника для размещения текста
        """
        # Левый верхний угол прямоугольника
        self.startX = self.centreX - cfg.valueWidth / 2
        self.startY = self.centreY - cfg.valueHeight / 2
        
        # Правый нижний угол прямоугольника
        self.endX = self.centreX + cfg.valueWidth / 2
        self.endY = self.centreY + cfg.valueHeight / 2
        
        # Точки для соединений с узлами (по бокам)
        self.leftX = self.startX
        self.leftY = self.centreY
        self.rightX = self.endX
        self.rightY = self.centreY
        
        # Центр прямоугольника для текста
        self.textX = (self.startX + self.endX) / 2
        self.textY = (self.startY + self.endY) / 2
    
    def create(self, canvas, coordinates, offset=0):
        """
        Создает визуальные элементы на холсте.
        
        Аргументы:
            canvas (tkinter.Canvas): холст, на котором будет отрисован блок
            coordinates (tuple): координаты (x, y) для размещения блока
            offset (int): смещение по оси X (по умолчанию 0)
        """
        # Сохраняем ссылку на холст
        self.canvas = canvas
        
        # Устанавливаем координаты центра с учетом смещения
        self.centreX, self.centreY = coordinates
        self.centreX += offset
        
        # Пересчитываем все координаты
        self.updatePoints()
        
        # Создаем прямоугольник (рамка блока)
        self.rid = self.canvas.create_rectangle(
            self.startX, self.startY,
            self.endX, self.endY,
            tags="valueBox"
        )
        
        # Закомментированный код: точки для соединений по бокам
        # Они могут понадобиться, если вы захотите рисовать линии от блока к узлам
        # self.startId = self.canvas.create_rectangle(
        #     self.leftX - cfg.dotHalfSize,
        #     self.leftY - cfg.dotHalfSize,
        #     self.leftX + cfg.dotHalfSize,
        #     self.leftY + cfg.dotHalfSize,
        #     fill="gray", tags="valueBox"
        # )
        # self.endId = self.canvas.create_rectangle(
        #     self.rightX - cfg.dotHalfSize,
        #     self.rightY - cfg.dotHalfSize,
        #     self.rightX + cfg.dotHalfSize,
        #     self.rightY + cfg.dotHalfSize,
        #     fill="gray", tags="valueBox"
        # )
        
        # Создаем текстовое поле с текущим значением
        # Форматируем число до 4 знаков после запятой для читаемости
        formatted_value = f"{self.value:.4f}"
        self.tId = self.canvas.create_text(
            self.textX,
            self.textY,
            text=formatted_value,
            tags="valueBox"
        )
    
    def setRightNode(self, node):
        """
        Устанавливает ссылку на правый узел (нейрон).
        
        Аргументы:
            node: объект узла (предположительно Node)
        """
        self.rightNode = node
    
    def setLeftNode(self, node):
        """
        Устанавливает ссылку на левый узел (нейрон).
        
        Аргументы:
            node: объект узла (предположительно Node)
        """
        self.leftNode = node
    
    def move(self, deltaX, deltaY):
        """
        Перемещает все элементы блока на заданное смещение.
        
        Аргументы:
            deltaX (int/float): смещение по горизонтали
            deltaY (int/float): смещение по вертикали
        """
        # Перемещаем прямоугольник
        self.canvas.move(self.rid, deltaX, deltaY)
        
        # Перемещаем текстовое поле
        self.canvas.move(self.tId, deltaX, deltaY)
        
        # Закомментированный код: перемещение точек соединений
        # self.canvas.move(self.endId, deltaX, deltaY)
        # self.canvas.move(self.startId, deltaX, deltaY)
        
        # Обновляем внутренние координаты для будущих операций
        self.centreX += deltaX
        self.centreY += deltaY
        self.updatePoints()
    
    def updateValue(self, new_value):
        """
        Обновляет отображаемое значение в текстовом поле.
        
        Аргументы:
            new_value (float): новое числовое значение
        """
        self.value = new_value
        formatted_value = f"{self.value:.4f}"
        self.canvas.itemconfig(self.tId, text=formatted_value)
    
    def setColor(self, color):
        """
        Устанавливает цвет фона прямоугольника.
        
        Аргументы:
            color (str): цвет в формате tkinter ("red", "#FF0000" и т.д.)
        """
        self.canvas.itemconfig(self.rid, fill=color)


# ==================== ТЕСТОВЫЙ ЗАПУСК (ДЛЯ ОТЛАДКИ) ====================
if __name__ == "__main__":
    """
    Этот блок выполняется только при запуске файла напрямую.
    Создает тестовое окно для проверки отображения ValueBox.
    """
    import tkinter as tk
    
    # Создаем тестовое окно
    test_window = tk.Tk()
    test_window.title("Тест ValueBox")
    test_window.geometry("400x300")
    
    # Создаем холст
    canvas = tk.Canvas(test_window, bg='white', width=400, height=300)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Создаем и отображаем тестовый блок ValueBox
    test_value = ValueBox()
    test_value.value = 0.12345
    test_value.create(canvas, (200, 150), 0)
    
    # Кнопка для обновления значения
    def update_random():
        import random
        new_val = random.uniform(-1, 1)
        test_value.updateValue(new_val)
        print(f"Значение обновлено: {new_val:.4f}")
    
    update_btn = tk.Button(
        test_window,
        text="Обновить значение",
        command=update_random
    )
    update_btn.pack(pady=10)
    
    # Запускаем тестовое окно
    test_window.mainloop()