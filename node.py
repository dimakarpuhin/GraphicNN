"""
Модуль node (Визуальное отображение нейронов/узлов)
Содержит класс Node, который создает и управляет графическим представлением
нейронов в нейросети на холсте.

Каждый объект Node представляет собой:
- Круг (овал) - визуальное представление нейрона
- Точки соединения по бокам (для связей с другими нейронами)
- Значение смещения (bias) над нейроном
- Выходное значение (value) под нейроном
- Связи с левыми и правыми стрелками (весами)
"""

import config as cfg
import random
import arrow as arr


class Node:
    """
    Класс для визуализации нейрона (узла) на Canvas.
    
    Атрибуты:
        bias (float): значение смещения для данного нейрона
        leftArrow (list): список левых связей (стрелки или блоки значений)
        rightArrow (list): список правых связей
        what (str): тип объекта (может быть "node")
        layer (int): номер слоя, к которому принадлежит нейрон
        value (float): выходное значение нейрона
        canvas: ссылка на холст tkinter.Canvas
        startX, startY: координаты левого верхнего угла ограничивающего прямоугольника
        endX, endY: координаты правого нижнего угла ограничивающего прямоугольника
        centerX, centerY: координаты центра круга
        leftX, leftY: координаты левой точки соединения
        rightX, rightY: координаты правой точки соединения
        cid: ID круга на Canvas
        startId: ID левой точки соединения
        endId: ID правой точки соединения
        tidbias: ID текстового поля для отображения смещения
        tidValue: ID текстового поля для отображения значения
    """
    
    def __init__(self):
        """
        Конструктор класса Node.
        Инициализирует новый узел со значениями по умолчанию.
        """
        # Смещение (bias) нейрона - будет установлено позже из контроллера
        self.bias = 0
        
        # Списки для хранения связей с другими элементами
        self.leftArrow = []   # Элементы слева от узла (стрелки или блоки значений)
        self.rightArrow = []  # Элементы справа от узла
        
        # Идентификационные атрибуты
        self.what = "node"     # Тип объекта
        self.layer = None      # Номер слоя в сети
        self.value = 0         # Выходное значение нейрона
        
        # Геометрические атрибуты (будут инициализированы в create())
        self.canvas = None
        self.startX = self.startY = 0
        self.endX = self.endY = 0
        self.centerX = self.centerY = 0
        self.leftX = self.leftY = 0
        self.rightX = self.rightY = 0
        self.textX = self.textY = 0
        self.textVX = self.textVY = 0
        
        # ID объектов на Canvas
        self.cid = None
        self.startId = None
        self.endId = None
        self.tidbias = None
        self.tidValue = None
    
    def updatePoints(self):
        """
        Пересчитывает координаты всех элементов узла на основе текущей позиции.
        Вызывается при создании узла и при его перемещении.
        
        Вычисляет:
            - endX, endY: правый нижний угол ограничивающего прямоугольника
            - centerX, centerY: центр круга
            - leftX, leftY: координаты левой точки соединения
            - rightX, rightY: координаты правой точки соединения
            - textX, textY: координаты для текста смещения (над узлом)
            - textVX, textVY: координаты для текста значения (под узлом)
        """
        # Правый нижний угол ограничивающего прямоугольника
        self.endX = self.startX + cfg.nodeWidth
        self.endY = self.startY + cfg.nodeHeight
        
        # Центр круга (нейрона)
        self.centerX = (self.startX + self.endX) / 2
        self.centerY = (self.startY + self.endY) / 2
        
        # Точки для соединений (по бокам круга)
        self.leftX = self.startX
        self.leftY = self.centerY
        self.rightX = self.endX
        self.rightY = self.centerY
        
        # Позиция текста смещения (над нейроном)
        self.textX = (self.startX + self.endX) / 2
        self.textY = (self.startY + self.endY) / 2 - cfg.textOffsetY
        
        # Позиция текста значения (под нейроном)
        self.textVX = (self.startX + self.endX) / 2
        self.textVY = (self.startY + self.endY) / 2 + cfg.textOffsetY
    
    def updateBias(self, bias_value):
        """
        Обновляет отображаемое значение смещения (bias).
        
        Аргументы:
            bias_value (float): новое значение смещения
            
        Примечание:
            - Положительные смещения отображаются красным цветом
            - Отрицательные или нулевые - зеленым
        """
        self.bias = bias_value
        self.canvas.itemconfig(self.tidbias, text=f"B: {self.bias:.4f}")
        
        # Цвет текста зависит от знака смещения
        if float(bias_value) <= 0:
            self.canvas.itemconfig(self.tidbias, fill="green")
        else:
            self.canvas.itemconfig(self.tidbias, fill="red")
    
    def updateValue(self, node_value):
        """
        Обновляет отображаемое выходное значение нейрона.
        
        Аргументы:
            node_value (float): новое значение нейрона
        """
        self.value = node_value
        self.canvas.itemconfig(self.tidValue, text=f"V: {self.value:.4f}")
    
    def create(self, canvas, x, y):
        """
        Создает визуальные элементы узла на холсте.
        
        Аргументы:
            canvas (tkinter.Canvas): холст для рисования
            x (int): X-координата левого верхнего угла
            y (int): Y-координата левого верхнего угла
        """
        # Сохраняем ссылку на холст и начальные координаты
        self.canvas = canvas
        self.startX = x
        self.startY = y
        
        # Пересчитываем все координаты
        self.updatePoints()
        
        # Создаем круг (тело нейрона) - оранжевого цвета, с тегом "drag" для перетаскивания
        self.cid = self.canvas.create_oval(
            self.startX, self.startY,
            self.endX, self.endY,
            fill="orange",
            tags="drag"
        )
        
        # Создаем левую точку соединения (маленький серый квадратик)
        self.startId = self.canvas.create_rectangle(
            self.leftX - cfg.dotHalfSize,
            self.leftY - cfg.dotHalfSize,
            self.leftX + cfg.dotHalfSize,
            self.leftY + cfg.dotHalfSize,
            fill="gray",
            tags="drag"
        )
        
        # Создаем правую точку соединения (маленький серый квадратик)
        self.endId = self.canvas.create_rectangle(
            self.rightX - cfg.dotHalfSize,
            self.rightY - cfg.dotHalfSize,
            self.rightX + cfg.dotHalfSize,
            self.rightY + cfg.dotHalfSize,
            fill="gray",
            tags="drag"
        )
        
        # Создаем текст для отображения смещения (над нейроном)
        self.tidbias = self.canvas.create_text(
            self.textX,
            self.textY,
            text=f"B: {self.bias:.4f}",
            fill="red",
            tag="drag"
        )
        
        # Создаем текст для отображения значения (под нейроном)
        self.tidValue = self.canvas.create_text(
            self.textVX,
            self.textVY,
            text=f"V: {self.value:.4f}",
            fill="blue",
            tag="drag"
        )
    
    def containAny(self, element_id):
        """
        Проверяет, принадлежит ли данный ID одному из элементов узла.
        
        Аргументы:
            element_id: ID элемента на Canvas
            
        Возвращает:
            bool: True если ID принадлежит узлу, False в противном случае
        """
        return element_id in (self.startId, self.endId, self.cid, self.tidbias, self.tidValue)
    
    def move(self, deltaX, deltaY):
        """
        Перемещает все элементы узла на заданное смещение.
        Также обновляет позиции всех связанных элементов (стрелок и блоков значений).
        
        Аргументы:
            deltaX (int/float): смещение по горизонтали
            deltaY (int/float): смещение по вертикали
        """
        # Перемещаем все графические элементы
        self.canvas.move(self.cid, deltaX, deltaY)
        self.canvas.move(self.endId, deltaX, deltaY)
        self.canvas.move(self.startId, deltaX, deltaY)
        self.canvas.move(self.tidbias, deltaX, deltaY)
        self.canvas.move(self.tidValue, deltaX, deltaY)
        
        # Обновляем внутренние координаты
        self.startX += deltaX
        self.startY += deltaY
        self.endX += deltaX
        self.endY += deltaY
        
        # Пересчитываем все зависимые координаты
        self.updatePoints()
        
        # Поднимаем элементы на передний план (чтобы они были видны)
        self.canvas.tag_raise(self.cid)
        self.canvas.tag_raise(self.startId)
        self.canvas.tag_raise(self.endId)
        self.canvas.tag_raise(self.tidbias)
        self.canvas.tag_raise(self.tidValue)
        
        # Перемещаем все связанные элементы слева
        for left_element in self.leftArrow:
            if left_element.what == "arrow":
                left_element.movePtRight(self.leftX, self.leftY)
            elif left_element.what == "valuebox":
                left_element.move(deltaX, deltaY)
        
        # Перемещаем все связанные элементы справа
        for right_element in self.rightArrow:
            if right_element.what == "arrow":
                right_element.movePtLeft(self.rightX, self.rightY)
            elif right_element.what == "valuebox":
                right_element.move(deltaX, deltaY)
    
    def getLeftCod(self):
        """
        Возвращает координаты левой точки соединения.
        
        Возвращает:
            tuple: (x, y) координаты левой точки
        """
        return (self.leftX, self.leftY)
    
    def getRightCod(self):
        """
        Возвращает координаты правой точки соединения.
        
        Возвращает:
            tuple: (x, y) координаты правой точки
        """
        return (self.rightX, self.rightY)
    
    def addLeftArrow(self, left_arrow):
        """
        Добавляет связь слева от узла.
        
        Аргументы:
            left_arrow: объект связи (стрелка или блок значения)
        """
        self.leftArrow.append(left_arrow)
    
    def addRightArrow(self, right_arrow):
        """
        Добавляет связь справа от узла.
        
        Аргументы:
            right_arrow: объект связи (стрелка или блок значения)
        """
        self.rightArrow.append(right_arrow)
    
    def setLayer(self, layer_number):
        """
        Устанавливает номер слоя для узла.
        
        Аргументы:
            layer_number (int): номер слоя в сети
        """
        self.layer = layer_number
    
    def setNodeColor(self, color):
        """
        Устанавливает цвет круга нейрона.
        
        Аргументы:
            color (str): цвет в формате tkinter ("red", "#FF0000" и т.д.)
        """
        self.canvas.itemconfig(self.cid, fill=color)


# ==================== ТЕСТОВЫЙ ЗАПУСК (ДЛЯ ОТЛАДКИ) ====================
if __name__ == "__main__":
    """
    Этот блок выполняется только при запуске файла напрямую.
    Создает тестовое окно для проверки отображения узлов.
    """
    import tkinter as tk
    
    # Создаем тестовое окно
    test_window = tk.Tk()
    test_window.title("Тест Node")
    test_window.geometry("400x400")
    
    # Создаем холст
    canvas = tk.Canvas(test_window, bg='white', width=400, height=400)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Создаем тестовый узел
    test_node = Node()
    test_node.create(canvas, 175, 175)
    test_node.updateBias(0.05)
    test_node.updateValue(0.75)
    
    # Информационная метка
    info_label = tk.Label(
        test_window,
        text="Нейрон: B - смещение (красный/зеленый), V - значение (синий)",
        bg='lightgray'
    )
    info_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Запускаем тестовое окно
    test_window.mainloop()