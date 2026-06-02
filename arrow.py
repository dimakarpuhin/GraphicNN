"""
Модуль arrow (Визуальное отображение стрелок/весов)
Содержит класс Arrow, который создает и управляет графическим представлением
связей (весов) между нейронами в нейросети на холсте.

Каждый объект Arrow представляет собой:
- Линию со стрелкой на конце (соединение между нейронами)
- Текстовое поле с значением веса (weight)
- Связи с левым и правым узлами (нейронами)

Цвет текста веса зависит от знака:
- Зеленый: отрицательный или нулевой вес
- Красный: положительный вес
"""

import config as cfg
import random


class Arrow:
    """
    Класс для визуализации связи (стрелки) между двумя нейронами на Canvas.
    
    Атрибуты:
        weight (float): значение веса для данной связи
        leftNode: ссылка на левый узел (откуда идет стрелка)
        rightNode: ссылка на правый узел (куда идет стрелка)
        what (str): тип объекта ("arrow")
        canvas: ссылка на холст tkinter.Canvas
        startX, startY: координаты начала стрелки (левый узел)
        endX, endY: координаты конца стрелки (правый узел)
        textX, textY: координаты для текста веса (середина линии)
        aid: ID линии стрелки на Canvas
        tid: ID текстового поля на Canvas
        startId: ID левой точки соединения (закомментирован)
        endId: ID правой точки соединения (закомментирован)
    """
    
    def __init__(self):
        """
        Конструктор класса Arrow.
        Инициализирует новую стрелку со значениями по умолчанию.
        """
        # Вес связи (будет установлен позже из контроллера)
        self.weight = 0.0
        
        # Ссылки на связанные узлы
        self.leftNode = None
        self.rightNode = None
        
        # Идентификационные атрибуты
        self.what = "arrow"
        
        # Геометрические атрибуты (будут инициализированы в create())
        self.canvas = None
        self.startX = self.startY = 0
        self.endX = self.endY = 0
        self.textX = self.textY = 0
        
        # ID объектов на Canvas
        self.aid = None      # ID линии (arrow id)
        self.tid = None      # ID текста (text id)
        
        # ID точек соединения (закомментированы, но оставлены для совместимости)
        self.startId = None
        self.endId = None
    
    def updateWeight(self, weight_value):
        """
        Обновляет отображаемое значение веса.
        
        Аргументы:
            weight_value (float): новое значение веса
            
        Примечание:
            - Положительные веса отображаются красным цветом
            - Отрицательные или нулевые - зеленым
        """
        self.weight = weight_value
        self.canvas.itemconfig(self.tid, text=f"W: {self.weight:.4f}")
        
        # Цвет текста зависит от знака веса
        if float(weight_value) <= 0:
            self.canvas.itemconfig(self.tid, fill="green")
        else:
            self.canvas.itemconfig(self.tid, fill="red")
    
    def updatePoints(self):
        """
        Пересчитывает координаты текста на основе текущих точек начала и конца.
        Вычисляет середину линии для размещения текста с весом.
        """
        self.textX = (self.startX + self.endX) / 2
        self.textY = (self.startY + self.endY) / 2
    
    def create(self, canvas, start_point, end_point):
        """
        Создает визуальные элементы стрелки на холсте.
        
        Аргументы:
            canvas (tkinter.Canvas): холст для рисования
            start_point (tuple): координаты (x, y) начала стрелки (левый узел)
            end_point (tuple): координаты (x, y) конца стрелки (правый узел)
        """
        # Сохраняем координаты
        self.startX, self.startY = start_point
        self.endX, self.endY = end_point
        
        # Пересчитываем позицию для текста
        self.updatePoints()
        
        # Сохраняем ссылку на холст
        self.canvas = canvas
        
        # Создаем линию со стрелкой на конце
        # arrow="last" означает, что стрелка будет на конце линии
        self.aid = self.canvas.create_line(
            self.startX, self.startY,
            self.endX, self.endY,
            arrow="last",
            tags="arrow"
        )
        
        # Закомментированный код: точки для перетаскивания на концах стрелки
        # Они могут понадобиться для ручного изменения формы связи
        # self.startId = self.canvas.create_rectangle(
        #     self.startX - cfg.dotHalfSize,
        #     self.startY - cfg.dotHalfSize,
        #     self.startX + cfg.dotHalfSize,
        #     self.startY + cfg.dotHalfSize,
        #     fill="gray",
        #     tags="arrow"
        # )
        # self.endId = self.canvas.create_rectangle(
        #     self.endX - cfg.dotHalfSize,
        #     self.endY - cfg.dotHalfSize,
        #     self.endX + cfg.dotHalfSize,
        #     self.endY + cfg.dotHalfSize,
        #     fill="gray",
        #     tags="arrow"
        # )
        
        # Создаем текст с значением веса
        # Используем шрифт из настроек (numberFont)
        self.tid = self.canvas.create_text(
            self.textX,
            self.textY,
            text=f"W: {self.weight:.4f}",
            tag="arrow",
            font=cfg.numberFont
        )
    
    def containArrowText(self, element_id):
        """
        Проверяет, принадлежит ли данный ID линии стрелки или тексту веса.
        
        Аргументы:
            element_id: ID элемента на Canvas
            
        Возвращает:
            bool: True если ID принадлежит стрелке или тексту, False в противном случае
        """
        return element_id in (self.aid, self.tid)
    
    def containPtStart(self, element_id):
        """
        Проверяет, является ли данный ID левой точкой соединения.
        
        Аргументы:
            element_id: ID элемента на Canvas
            
        Возвращает:
            bool: True если ID соответствует левой точке
        """
        return element_id == self.startId
    
    def containPtEnd(self, element_id):
        """
        Проверяет, является ли данный ID правой точкой соединения.
        
        Аргументы:
            element_id: ID элемента на Canvas
            
        Возвращает:
            bool: True если ID соответствует правой точке
        """
        return element_id == self.endId
    
    def moveArrowText(self, deltaX, deltaY):
        """
        Перемещает все элементы стрелки на заданное смещение.
        
        Аргументы:
            deltaX (int/float): смещение по горизонтали
            deltaY (int/float): смещение по вертикали
        """
        # Перемещаем линию
        self.canvas.move(self.aid, deltaX, deltaY)
        
        # Перемещаем текст
        self.canvas.move(self.tid, deltaX, deltaY)
        
        # Обновляем внутренние координаты
        self.startX += deltaX
        self.startY += deltaY
        self.endX += deltaX
        self.endY += deltaY
        
        # Пересчитываем позицию текста
        self.updatePoints()
    
    def movePtLeft(self, x, y):
        """
        Перемещает левую точку стрелки (начало) в новые координаты.
        Используется, когда левый узел перемещается.
        
        Аргументы:
            x (int/float): новая X-координата левой точки
            y (int/float): новая Y-координата левой точки
        """
        # Обновляем координаты начала
        self.startX, self.startY = x, y
        
        # Обновляем координаты линии
        self.canvas.coords(self.aid, self.startX, self.startY, self.endX, self.endY)
        
        # Пересчитываем позицию текста
        self.updatePoints()
        
        # Обновляем позицию текста
        self.canvas.coords(self.tid, self.textX, self.textY)
    
    def movePtRight(self, x, y):
        """
        Перемещает правую точку стрелки (конец) в новые координаты.
        Используется, когда правый узел перемещается.
        
        Аргументы:
            x (int/float): новая X-координата правой точки
            y (int/float): новая Y-координата правой точки
        """
        # Обновляем координаты конца
        self.endX, self.endY = x, y
        
        # Обновляем координаты линии
        self.canvas.coords(self.aid, self.startX, self.startY, self.endX, self.endY)
        
        # Пересчитываем позицию текста
        self.updatePoints()
        
        # Обновляем позицию текста
        self.canvas.coords(self.tid, self.textX, self.textY)
    
    def changeDefaultColor(self):
        """
        Восстанавливает цвета по умолчанию для элементов стрелки.
        (Для совместимости с кодом, где есть точки соединения)
        """
        if self.endId:
            self.canvas.itemconfig(self.endId, fill="gray")
        if self.startId:
            self.canvas.itemconfig(self.startId, fill="gray")
        if self.aid:
            self.canvas.itemconfig(self.aid, fill="black")
    
    def setLeftNode(self, node):
        """
        Устанавливает ссылку на левый узел.
        
        Аргументы:
            node: объект Node слева
        """
        self.leftNode = node
    
    def setRightNode(self, node):
        """
        Устанавливает ссылку на правый узел.
        
        Аргументы:
            node: объект Node справа
        """
        self.rightNode = node


# ==================== ТЕСТОВЫЙ ЗАПУСК (ДЛЯ ОТЛАДКИ) ====================
if __name__ == "__main__":
    """
    Этот блок выполняется только при запуске файла напрямую.
    Создает тестовое окно для проверки отображения стрелок.
    """
    import tkinter as tk
    
    # Создаем тестовое окно
    test_window = tk.Tk()
    test_window.title("Тест Arrow")
    test_window.geometry("500x300")
    
    # Создаем холст
    canvas = tk.Canvas(test_window, bg='white', width=500, height=300)
    canvas.pack(fill=tk.BOTH, expand=True)
    
    # Создаем тестовую стрелку от (100,150) до (400,150)
    test_arrow = Arrow()
    test_arrow.create(canvas, (100, 150), (400, 150))
    test_arrow.updateWeight(0.75)
    
    # Кнопка для изменения веса
    def random_weight():
        import random
        new_weight = random.uniform(-1, 1)
        test_arrow.updateWeight(new_weight)
        print(f"Вес обновлен: {new_weight:.4f}")
    
    # Кнопка для изменения цвета стрелки
    def change_color():
        if test_arrow.weight <= 0:
            canvas.itemconfig(test_arrow.aid, fill="green")
        else:
            canvas.itemconfig(test_arrow.aid, fill="red")
    
    # Панель с кнопками
    button_frame = tk.Frame(test_window)
    button_frame.pack(side=tk.BOTTOM, pady=10)
    
    random_btn = tk.Button(button_frame, text="Случайный вес", command=random_weight)
    random_btn.pack(side=tk.LEFT, padx=5)
    
    color_btn = tk.Button(button_frame, text="Показать цвет стрелки", command=change_color)
    color_btn.pack(side=tk.LEFT, padx=5)
    
    # Информационная метка
    info_label = tk.Label(
        test_window,
        text="Стрелка: W - вес (красный/зеленый)", 
        bg='lightgray'
    )
    info_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Рисуем два круга для наглядности (имитация нейронов)
    left_circle = canvas.create_oval(80, 130, 120, 170, fill="orange", outline="black")
    right_circle = canvas.create_oval(380, 130, 420, 170, fill="orange", outline="black")
    
    # Подписи к нейронам
    canvas.create_text(100, 200, text="Левый нейрон", font=("Arial", 10))
    canvas.create_text(400, 200, text="Правый нейрон", font=("Arial", 10))
    
    # Запускаем тестовое окно
    test_window.mainloop()