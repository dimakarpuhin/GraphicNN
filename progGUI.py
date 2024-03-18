import tkinter as tk
from tkinter import ttk
from tkinter import *

import pUtility as pu



class View: 

    """Метод инициализации объектов."""
    def __init__(self, parent): #  
        # инициализировать переменные
        self.container = parent # род. элемент - заданная точка входа в GUI

    """Метод для настройки пользовательского интерфейса"""
    def setup(self): 
        
        self.create_widgets() # метод по созданию виджитов
        self.setup_layout() # метод по настройки виджетов внутри макетов и инициализации значений

        """Метод создания различных виджетов в главном окне tkinter."""
    def create_widgets(self):
        
        # Создать 3 рамки, слева по середине и справа (называются Frame по англ.)
        self.leftFrame = Frame(self.container,width=100, height=pu.windowHeight)
        self.midFrame = Frame(self.container,width=pu.canvasWidth, height=pu.canvasHeight)
        self.rightFrame = Frame(self.container,width=100, height=pu.windowHeight)
        self.rightFrame1=Frame(self.rightFrame) # верхнее окно внутри правого рамки
        self.rightFrame2=Frame(self.rightFrame) # нижнее окно внутри правого рамки


        # Строка состояния
        self.statusbar=tk.Label(self.midFrame,text="On the way…", bd=1, relief=tk.SUNKEN, anchor=tk.W,fg="red")

        # Виджеты по созданию аохитектуры нейросети внутри левой рамки.
        self.lab1input = tk.Label(self.leftFrame,text="Входной слой")
        self.combo1input = ttk.Combobox(self.leftFrame,values=[1,2,3,4])
        self.lab2layer1 = tk.Label(self.leftFrame, text="Скрытый слой 1")
        self.combo2layer1 = ttk.Combobox(self.leftFrame, values=[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.lab3layer2 = tk.Label(self.leftFrame, text="Скрытый слой 2")
        self.combo3layer2 = ttk.Combobox(self.leftFrame, values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.lab4layer3 = tk.Label(self.leftFrame, text="Скрытый слой 3")
        self.combo4layer3 = ttk.Combobox(self.leftFrame, values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.lab5output = tk.Label(self.leftFrame, text="Выходной слой")
        self.combo5output = ttk.Combobox(self.leftFrame, values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Виджет для кнопки "Создать нейросеть" в левом фрейме.
        self.b2Circle = tk.Button(self.leftFrame, text = "Создать нейросеть", width =20, height =1)

        # Виджеты для создания пользовательского интерфейса по обучению НС внутри левой рамки.
        self.lab6TrainDiv = tk.Label(self.leftFrame, text="===Обучение===") # Надпись "Обучение" для отделения групп
        self.lab7LearningRate=tk.Label(self.leftFrame,text="Скорость обучения")
        self.combo6learningRate=ttk.Combobox(self.leftFrame,values=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5])
        self.lab8epoch=tk.Label(self.leftFrame,text="Количество эпох")
        self.combo7epoch = ttk.Combobox(self.leftFrame,
                                                values=[10,5000,10000,50000,100000,200000,500000,1000000,2000000,5000000,
                                                        10000000,20000000,50000000])
        self.lab9refreshRate=tk.Label(self.leftFrame,text="Частота обновления(эпох)")
        self.combo8refreshRate = ttk.Combobox(self.leftFrame,
                                        values=[2,100,200,300,500,1000,2000,3000])

        # Виджеты для 3 кнопок по обучению НС внутри левой рамки.
        self.b8LoadFeature=tk.Button(self.leftFrame, text="Загрузить хар-ки НС", width=20, height=1) 
        self.b9LoadLabel = tk.Button(self.leftFrame, text="Загрузить метки НС", width=20, height=1) 
        self.b10StartTrain=tk.Button(self.leftFrame, text="Начать обучение", width=20, height=1)

        # Виджеты для загрузки и сохранения НС внутри левой рамки.
        self.lab9aSaveLoadFile = tk.Label(self.leftFrame, text="===Сохранить/Загрузить===")
        self.b10aSaveNN = tk.Button(self.leftFrame, text="Сохранить нейросеть", width=20, height=1)
        self.b10bLoadNN = tk.Button(self.leftFrame, text="Загрузить нейросеть", width=20, height=1)

        # Секция/Виджеты для прогнозирование НС внутри левой рамки.
        self.lab10DivPredection=tk.Label(self.leftFrame,text="===Прогнозирование===")
        self.b11LoadPreFeature = tk.Button(self.leftFrame, text="Загрузить ф-цию прогноз.", width=20,height=1) 
        self.b12StartPredict = tk.Button(self.leftFrame, text="Начать прогнозирование", width=20, height=1)
        self.lab11PreRounding=tk.Label(self.leftFrame,text="===Округление===")
        self.combo9PreRounding = ttk.Combobox(self.leftFrame, values=[1, 0.1, 0.01, 0.001, 0.0001, 0.00001])

        # Параметры центрального окна(рамки) его цвет, ширина, высота
        self.c = Canvas(self.midFrame, bg='white', width=pu.canvasWidth, height=pu.canvasHeight)

        # Текстовые поля частоты ошибок и прогнозирования внутри правой рамки
        self.lab31ErrorRate=tk.Label(self.rightFrame,text="===Функция потерь НС===")
        self.text31ErrorTxtbox=tk.Text(self.rightFrame1,width=30)
        self.scl31vbar = tk.Scrollbar(self.rightFrame1, orient=VERTICAL) # полоса прокрутки
        self.text31ErrorTxtbox.config(yscrollcommand=self.scl31vbar.set)
        self.scl31vbar.config(command=self.text31ErrorTxtbox.yview)

        self.lab32PredictionResult=tk.Label(self.rightFrame,text="===Результат предсказания===")
        self.text32PredictTxtbox = tk.Text(self.rightFrame2,width=30)
        self.scl32vbarPre = tk.Scrollbar(self.rightFrame2, orient=VERTICAL)
        self.text32PredictTxtbox.config(yscrollcommand=self.scl32vbarPre.set)
        self.scl32vbarPre.config(command=self.text32PredictTxtbox.yview)
    
    """Метод установки расположения виджетов в окнах."""
    def setup_layout(self):

        # Установка 3 главных рамок
        self.leftFrame.pack(side = LEFT)
        self.midFrame.pack (side = LEFT)
        self.rightFrame.pack (side = RIGHT)

        # Установка раздела по созданию НС вверху левой рамки
        self.lab1input.pack(side=TOP)
        self.combo1input.pack(side=TOP)
        self.lab2layer1.pack(side=TOP)
        self.combo2layer1.pack(side=TOP)
        self.lab3layer2.pack(side=TOP)
        self.combo3layer2.pack(side=TOP)
        self.lab4layer3.pack(side=TOP)
        self.combo4layer3.pack(side=TOP)
        self.lab5output.pack(side=TOP)
        self.combo5output.pack(side=TOP)
        self.b2Circle.pack(side = TOP)

        # Установка раздела обучение в левой рамке
        self.lab6TrainDiv.pack(side=TOP)
        self.lab7LearningRate.pack(side=TOP)
        self.combo6learningRate.pack(side=TOP)
        self.lab8epoch.pack(side=TOP)
        self.combo7epoch.pack(side=TOP)
        self.lab9refreshRate.pack(side=TOP)
        self.combo8refreshRate.pack(side=TOP)
        self.b8LoadFeature.pack(side=TOP)
        self.b9LoadLabel.pack(side=TOP)
        self.b10StartTrain.pack(side=TOP)

        # Установка раздела сохр/загр НС в левой рамке
        self.lab9aSaveLoadFile.pack(side=TOP)
        self.b10aSaveNN.pack(side=TOP)
        self.b10bLoadNN.pack(side=TOP)

        # Установка раздела предсказания в левой рамке
        self.lab10DivPredection.pack(side=TOP)
        self.b11LoadPreFeature.pack(side=TOP)
        self.b12StartPredict.pack(side=TOP)
        self.lab11PreRounding.pack(side=TOP)
        self.combo9PreRounding.pack(side=TOP)

        # Установка раздела ф-ции потерь в правой рамке
        self.lab31ErrorRate.pack(side=TOP)
        self.rightFrame1.pack(side=TOP, fill=BOTH)
        self.text31ErrorTxtbox.pack(side=LEFT)
        self.scl31vbar.pack(side=LEFT,fill=Y)

        # Установка раздела результат предсказания в правой рамке
        self.lab32PredictionResult.pack(side=TOP)
        self.rightFrame2.pack(side=BOTTOM, fill=BOTH)
        self.text32PredictTxtbox.pack(side=LEFT)
        self.scl32vbarPre.pack(side=LEFT,fill=Y)

        # Установить центральную рамку
        self.c.pack(side=TOP)

        # Установка раздела статус внизу центральной рамки
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Установить в полях создания НС значения по умолчанию 
        self.combo1input.current(1) # Входной слой
        self.combo2layer1.current(3) # 1 скрытый слой
        self.combo3layer2.current(0) # 2 скрытый слой
        self.combo4layer3.current(0) # 3 скрытый слой
        self.combo5output.current(2) # выходной слой

        # Уставки значений по умолчанию в секции "Тренировка"
        self.combo6learningRate.current(2) # скорость обучения
        self.combo7epoch.current(1)#1, # количесвто эпох
        self.combo8refreshRate.current(2) # частота обновления эпох
        self.combo9PreRounding.current(1) # округление по умолчанию 0.1

        # Сообщение в разделе статус
        self.statusbar["text"] = "Готов."








#Тестовое окно просмотра
if __name__ == "__main__": 
    mainwin = Tk()
    WIDTH = pu.windowWidth
    HEIGHT = pu.windowHeight
    mainwin.geometry("%sx%s" % (WIDTH, HEIGHT))
    mainwin.title("Тестовое окно")

    
    view=View(mainwin) # получить ссылку на объект просмотра
    view.setup() # после вызываем метод для настройки пользовательского интерфейса  
    mainwin.mainloop() # запустить главный цикл, чтобы окно могло реагировать на дейтвия пользователя