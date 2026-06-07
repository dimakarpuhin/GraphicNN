"""
Модуль neural_network (Нейросеть) для приложения NeuroCanvas
Реализует бизнес-логику: создание нейросети, обучение, прогнозирование.

В паттерне MVC этот модуль выступает в роли контроллера, управляя:
- Моделью (нейронная сеть, веса, смещения)
- Взаимодействием с представлением (View, графический интерфейс)

Основные функции:
- Создание и инициализация нейросети случайными весами и смещениями
- Обучение методом обратного распространения ошибки (backpropagation)
- Прогнозирование на основе обученной сети
- Загрузка обучающих/тестовых данных из CSV файлов
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import numpy as np
import csv

# Импорт пользовательских модулей
import config as cfg
from pubsub import pub


class NeuralNetwork:
    """
    Класс нейронной сети (контроллер).
    Управляет нейросетью, обрабатывает действия пользователя из View.
    
    Атрибуты:
        view: ссылка на объект View (графический интерфейс)
        numInput: количество входных нейронов
        numOutput: количество выходных нейронов
        numOfHLayers: список с количеством нейронов в скрытых слоях
        listWeight: список матриц весов для каждого слоя
        listFlatWeight: плоский список всех весов (для отображения)
        listBias: список векторов смещений для каждого слоя
        listFlatBias: плоский список всех смещений
        trainFeatureSet: обучающие признаки (входные данные)
        testFeatureSet: тестовые признаки
        trainLabelSet: обучающие метки (целевые значения)
        error_cost: список ошибок по эпохам
        ao: выход сети после softmax (предсказания)
    """
    
    def __init__(self):
        """Конструктор. Инициализирует все переменные."""
        
        # Связь с представлением
        self.view = None
        
        # Параметры архитектуры нейросети
        self.numInput = 0
        self.numOutput = 0
        self.numOfHLayers = []      # Количество нейронов в скрытых слоях
        
        # Веса и смещения
        self.listWeight = []        # Матрицы весов для каждого слоя
        self.listFlatWeight = []    # Уплощенные веса (для отображения на Canvas)
        self.listBias = []          # Векторы смещений
        self.listFlatBias = []      # Уплощенные смещения
        
        # Данные для обучения и тестирования
        self.trainFeatureSet = []
        self.testFeatureSet = []
        self.trainLabelSet = []
        
        # Флаги загрузки данных
        self.flagTrainFeatureLoad = False
        self.flagTestFeatureLoad = False
        self.flagTrainLabelLoad = False
        
        # Промежуточные значения для обратного распространения
        self.listAhiddenLayer = []  # Значения после сигмоиды (активации)
        self.listZhiddenLayer = []  # Суммы до сигмоиды
        self.error_cost = []        # История ошибок
        
        self.ao = None              # Выход сети (после softmax)
        
        # Пути к файлам
        self.inputTrainFilePath = ""
        self.outputTrainFilePath = ""
        
        # Фиксируем seed для воспроизводимости результатов
        np.random.seed(42)
    
    # ==================== СВЯЗЬ С ПРЕДСТАВЛЕНИЕМ ====================
    
    def setView(self, view):
        """
        Устанавливает связь между контроллером и представлением.
        
        Аргументы:
            view: экземпляр класса View
        """
        self.view = view
    
    # ==================== ИНИЦИАЛИЗАЦИЯ ВЕСОВ И СМЕЩЕНИЙ ====================
    
    def randomBias(self, listNodeValues):
        """
        Генерирует случайные смещения (bias) для каждого слоя.
        
        Аргументы:
            listNodeValues: список списков, где каждый подсписок содержит узлы слоя
        """
        self.listBias = []
        
        for i in range(1, len(listNodeValues)):
            currentNodeQty = len(listNodeValues[i])
            npBias = np.random.rand(currentNodeQty)
            self.listBias.append(npBias)
        
        self.flattenListBias(self.listBias)
        if self.view:
            self.view.updateBias(self.listFlatBias)
    
    def randomWeights(self, listNodeValues):
        """
        Генерирует случайные веса (weights) между слоями.
        
        Аргументы:
            listNodeValues: список списков, каждый подсписок содержит узлы слоя
        """
        self.listWeight = []
        
        for i in range(len(listNodeValues) - 1):
            currentNodeQty = len(listNodeValues[i])
            nextNodeQty = len(listNodeValues[i + 1])
            # Создаем матрицу весов размером [текущий_слой x следующий_слой]
            npMatrixWeight = np.random.rand(currentNodeQty, nextNodeQty)
            self.listWeight.append(npMatrixWeight)
        
        # Уплощаем для отображения в View
        self.flattenListWeight(self.listWeight)
        if self.view:
            self.view.updateWeight(self.listFlatWeight)
    
    def flattenListBias(self, biasList):
        """
        Преобразует список списков смещений в плоский список.
        
        Аргументы:
            biasList: список смещений по слоям
        """
        self.listFlatBias = []
        for layer in biasList:
            for i in layer:
                self.listFlatBias.append(i)
    
    def flattenListWeight(self, weightList):
        """
        Преобразует список матриц весов в плоский список.
        
        Аргументы:
            weightList: список матриц весов по слоям
        """
        self.listFlatWeight = []
        for layer in weightList:
            for i in layer:
                for j in i:
                    self.listFlatWeight.append(j)
        
        cfg.tprint(f"Длина плоского списка весов: {len(self.listFlatWeight)}")
    
    # ==================== АКТИВАЦИОННЫЕ ФУНКЦИИ ====================
    
    def sigmoid(self, x):
        """
        Сигмоидальная функция активации.
        Преобразует входное значение в диапазон (0, 1).
        
        Аргументы:
            x: входное значение (скаляр или numpy массив)
            
        Возвращает:
            результат сигмоиды
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_der(self, x):
        """
        Производная сигмоидальной функции.
        Используется для обратного распространения ошибки.
        
        Аргументы:
            x: входное значение
            
        Возвращает:
            производную сигмоиды
        """
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def softmax(self, A):
        """
        Функция Softmax для выходного слоя.
        Преобразует выходные значения в вероятности (сумма = 1).
        
        Аргументы:
            A: входной массив значений
            
        Возвращает:
            массив вероятностей
        """
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)
    
    # ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================
    
    def strToNp(self, stringList):
        """
        Преобразует список строк в список чисел с плавающей точкой (numpy float64).
        
        Аргументы:
            stringList: список строковых представлений чисел
            
        Возвращает:
            список чисел float64
        """
        newList = []
        for item in stringList:
            newList.append(np.float64(item))
        return newList
    
    def clear(self):
        """Очищает все данные о текущей нейросети (сброс состояния)."""
        self.numInput = 0
        self.numOutput = 0
        self.numOfHLayers.clear()
        self.listWeight.clear()
        self.listFlatWeight.clear()
        self.listBias.clear()
        self.listFlatBias.clear()
        # Данные обучения не очищаем, только архитектуру сети
    
    # ==================== ПРОГНОЗИРОВАНИЕ ====================
    
    def startPredict(self):
        """
        Запускает процесс прогнозирования на тестовых данных.
        Отображает результаты в текстовом поле view.text32PredictTxtbox.
        """
        startTime = datetime.now()
        
        # Очищаем поле результатов
        self.view.text32PredictTxtbox.delete('1.0', tk.END)
        
        # Выполняем прогнозирование
        result = self.predictLoop()
        
        cfg.tprint("Результат прогнозирования:")
        cfg.tprint(result)
        
        # Получаем настройку округления
        roundingIndex = self.view.combo9PreRounding.current()
        roundingValues = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        rounding = roundingValues[roundingIndex] if roundingIndex < len(roundingValues) else 1
        
        # Форматируем и выводим результаты
        for row in result:
            line = ""
            for i, value in enumerate(row):
                # Округляем в соответствии с настройкой
                if '.' in str(rounding):
                    precision = len(str(rounding).split('.')[-1])
                    roundedValue = round(value, precision)
                else:
                    roundedValue = round(value, 0)
                
                if i != len(row) - 1:
                    line += f"{roundedValue},"
                else:
                    line += f"{roundedValue}\n"
            
            self.view.text32PredictTxtbox.insert(tk.INSERT, line)
        
        # Выводим время выполнения
        runTime = datetime.now() - startTime
        self.view.statusbar["text"] = f"Прогнозирование завершено. Время выполнения: {runTime}"
    
    def predictLoop(self):
        """
        Выполняет прямой проход (forward pass) для прогнозирования.
        
        Возвращает:
            выход сети (предсказанные значения)
        """
        previous_set = self.testFeatureSet  # Начинаем с входных данных
        
        for i in range(len(self.listWeight)):
            if i != len(self.listWeight) - 1:
                # Скрытые слои: сумма -> сигмоида
                zh = np.dot(previous_set, self.listWeight[i]) + self.listBias[i]
                ah = self.sigmoid(zh)
                previous_set = ah
            else:
                # Выходной слой: сумма -> softmax
                zo = np.dot(previous_set, self.listWeight[i]) + self.listBias[i]
                ao = self.softmax(zo)
        
        return ao
    
    # ==================== ОБУЧЕНИЕ ====================
    
    def startTrain(self):
        """
        Запускает процесс обучения нейросети.
        Отображает функцию потерь в текстовом поле view.text31ErrorTxtbox.
        """
        startTime = datetime.now()
        
        # Очищаем поле ошибок
        self.view.text31ErrorTxtbox.delete('1.0', tk.END)
        
        # Получаем параметры обучения из интерфейса
        epochs = int(self.view.combo7epoch.get())
        refreshRate = int(self.view.combo8refreshRate.get())
        
        totalCheck = epochs // refreshRate
        progressCounter = 1
        
        for epoch in range(epochs):
            # Одна эпоха обучения
            self.trainLoop()
            
            # Обновляем отображение каждые refreshRate эпох
            if epoch % refreshRate == 0:
                # Вычисляем функцию потерь (кросс-энтропия)
                loss = np.sum(-np.array(self.trainLabelSet) * np.log(self.ao + 1e-10))  # Добавлен эпсилон для стабильности
                self.view.text31ErrorTxtbox.insert(tk.INSERT, f"{loss}\n")
                self.view.text31ErrorTxtbox.update()
                
                # Обновляем статус с процентом выполнения
                if totalCheck > 0:
                    progressPercent = progressCounter / totalCheck * 100
                else:
                    progressPercent = 100
                    
                runTime = datetime.now() - startTime
                self.view.statusbar["text"] = (
                    f"Обучение: {int(progressPercent)}% завершено. "
                    f"Время: {runTime}"
                )
                progressCounter += 1
                
                # Обновляем отображение весов и смещений в View
                self.flattenListWeight(self.listWeight)
                if self.view:
                    self.view.updateWeight(self.listFlatWeight)
                
                self.flattenListBias(self.listBias)
                if self.view:
                    self.view.updateBias(self.listFlatBias)
        
        runTime = datetime.now() - startTime
        self.view.statusbar["text"] = f"Обучение завершено. Время выполнения: {runTime}"
    
    def trainLoop(self):
        """
        Одна эпоха обучения (один проход по всем обучающим данным).
        Реализует прямое распространение и обратное распространение ошибки.
        """
        
        # ==================== ПРЯМОЕ РАСПРОСТРАНЕНИЕ (FORWARD PASS) ====================
        previous_set = self.trainFeatureSet
        self.listAhiddenLayer.clear()
        self.listZhiddenLayer.clear()
        
        for i in range(len(self.listWeight)):
            if i != len(self.listWeight) - 1:
                # Скрытые слои
                zh = np.dot(previous_set, self.listWeight[i]) + self.listBias[i]
                ah = self.sigmoid(zh)
                self.listAhiddenLayer.append(ah)
                self.listZhiddenLayer.append(zh)
                previous_set = ah
            else:
                # Выходной слой
                zo = np.dot(previous_set, self.listWeight[i]) + self.listBias[i]
                self.ao = self.softmax(zo)
                self.listAhiddenLayer.append(self.ao)
                self.listZhiddenLayer.append(zo)
        
        # ==================== ОБРАТНОЕ РАСПРОСТРАНЕНИЕ (BACKWARD PASS) ====================
        targetValue_set = self.trainLabelSet
        value_set = self.ao
        lr = float(self.view.combo6learningRate.get())  # Скорость обучения
        
        for i in range(len(self.listWeight) - 1, -1, -1):
            if i == len(self.listWeight) - 1:
                # Выходной слой
                dcost_dzo = value_set - targetValue_set
                dzo_dwo = self.listAhiddenLayer[i - 1]
                dcost_wo = np.dot(dzo_dwo.T, dcost_dzo)  # Корректировка весов
                dcost_bo = dcost_dzo  # Корректировка смещений
                
                # Сохраняем неизмененными для следующей итерации
                unchangedWeight = self.listWeight[i].copy()
                
                # Обновляем веса и смещения
                self.listWeight[i] -= lr * dcost_wo
                self.listBias[i] -= lr * dcost_bo.sum(axis=0)
            
            else:
                # Скрытые слои
                dzo_dah = unchangedWeight
                dcost_dah = np.dot(dcost_dzo, dzo_dah.T)
                dah_dzh = self.sigmoid_der(self.listZhiddenLayer[i])
                
                if i == 0:
                    previous_input_set = self.trainFeatureSet
                else:
                    previous_input_set = self.listAhiddenLayer[i - 1]
                
                dzh_dwh = np.array(previous_input_set)
                dcost_wh = np.dot(dzh_dwh.T, dah_dzh * dcost_dah)
                dcost_bh = dcost_dah * dah_dzh
                
                # Обновляем dcost_dzo для следующей итерации
                dcost_dzo = dcost_bh
                
                # Сохраняем для следующего слоя
                unchangedWeight = self.listWeight[i].copy()
                
                # Обновляем веса и смещения
                self.listWeight[i] -= lr * dcost_wh
                self.listBias[i] -= lr * dcost_bh.sum(axis=0)
    
    # ==================== ЗАГРУЗКА ДАННЫХ ====================
    
    def loadTestFeature(self, filePath):
        """
        Загружает тестовые признаки из CSV файла.
        
        Аргументы:
            filePath: путь к файлу с данными
        """
        if len(filePath) > 0:
            self.view.statusbar["text"] = "Загрузка тестовых данных..."
            
            dataList = []
            with open(filePath, "r") as file:
                lines = file.read().split('\n')
            
            for line in lines:
                if line != "":
                    row = self.strToNp(line.split(","))
                    dataList.append(row)
            
            self.testFeatureSet = dataList
            
            # Проверяем совместимость с входным слоем
            if len(self.testFeatureSet[0]) != len(self.view.listInputNode):
                messagebox.showinfo(
                    title="Ошибка",
                    message="Набор данных не совместим с архитектурой нейросети"
                )
                self.view.statusbar["text"] = "Загрузка тестовых данных не удалась"
                return
            
            self.setInputValue(self.testFeatureSet)
            self.flagTestFeatureLoad = True
            self.view.statusbar["text"] = "Тестовые данные загружены"
        else:
            self.flagTestFeatureLoad = False
            self.testFeatureSet.clear()
            self.view.clearInputValue()
            self.view.statusbar["text"] = "Загрузка тестовых данных не удалась"
    
    def loadTrainFeature(self, filePath):
        """
        Загружает обучающие признаки из CSV файла.
        
        Аргументы:
            filePath: путь к файлу с данными
        """
        if len(filePath) > 0:
            self.view.statusbar["text"] = "Загрузка обучающих данных..."
            
            dataList = []
            with open(filePath, "r") as file:
                lines = file.read().split('\n')
            
            for line in lines:
                if line != "":
                    row = self.strToNp(line.split(","))
                    dataList.append(row)
            
            self.trainFeatureSet = dataList
            
            # Если размерность не совпадает, обновляем архитектуру
            if len(self.trainFeatureSet[0]) != len(self.view.listInputNode):
                self.view.combo1input.set(len(self.trainFeatureSet[0]))
                self.view.createNodes()
            
            self.setInputValue(self.trainFeatureSet)
            self.flagTrainFeatureLoad = True
            self.view.statusbar["text"] = "Обучающие данные загружены"
        else:
            self.flagTrainFeatureLoad = False
            self.trainFeatureSet.clear()
            self.view.clearInputValue()
            self.view.statusbar["text"] = "Загрузка обучающих данных не удалась"
    
    def setInputValue(self, dataSet):
        """
        Устанавливает значения входных узлов для отображения в View.
        
        Аргументы:
            dataSet: набор данных (первая строка используется для отображения)
        """
        for i in range(len(dataSet[0])):
            self.view.listInputNode[i].updateValue(dataSet[0][i])
    
    def loadTrainLabel(self, filePath):
        """
        Загружает обучающие метки из CSV файла.
        
        Аргументы:
            filePath: путь к файлу с метками
        """
        if len(filePath) > 0:
            self.outputTrainFilePath = filePath
            self.view.statusbar["text"] = "Загрузка обучающих меток..."
            
            dataList = []
            with open(filePath, "r") as file:
                lines = file.read().split('\n')
            
            for line in lines:
                if line != "":
                    cfg.tprint(line)
                    row = self.strToNp(line.split(","))
                    dataList.append(row)
            
            self.trainLabelSet = dataList
            cfg.tprint(f"Обучающие метки: {self.trainLabelSet}")
            
            # Если размерность не совпадает, обновляем архитектуру
            if len(self.trainLabelSet[0]) != len(self.view.listOutputNode):
                self.view.combo5output.set(len(self.trainLabelSet[0]))
                self.view.createNodes()
                
                # Восстанавливаем значения входных узлов
                for i in range(len(self.trainFeatureSet[0])):
                    self.view.listInputNode[i].updateValue(self.trainFeatureSet[0][i])
            
            # Отображаем значения выходных узлов
            for i in range(len(self.trainLabelSet[0])):
                self.view.listOutputNode[i].updateValue(self.trainLabelSet[0][i])
            
            self.flagTrainLabelLoad = True
            self.view.statusbar["text"] = "Обучающие метки загружены"
        else:
            self.flagTrainLabelLoad = False
            self.trainLabelSet.clear()
            self.view.clearOutputValue()
            self.view.statusbar["text"] = "Загрузка обучающих меток не удалась"
    
    # ==================== ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ ====================
    
    def create_network(self):
        """Создает нейросеть на основе выбранных параметров."""
        # Получаем параметры из интерфейса
        input_size = int(self.view.combo1input.get())
        hidden1 = int(self.view.combo2layer1.get())
        hidden2 = int(self.view.combo3layer2.get())
        hidden3 = int(self.view.combo4layer3.get())
        output_size = int(self.view.combo5output.get())
        
        # Формируем структуру сети
        self.numOfHLayers = []
        if hidden1 > 0:
            self.numOfHLayers.append(hidden1)
        if hidden2 > 0:
            self.numOfHLayers.append(hidden2)
        if hidden3 > 0:
            self.numOfHLayers.append(hidden3)
        
        self.numInput = input_size
        self.numOutput = output_size
        
        # Создаем список узлов для инициализации весов
        listNodeValues = [[0] * input_size]
        for h in self.numOfHLayers:
            listNodeValues.append([0] * h)
        listNodeValues.append([0] * output_size)
        
        # Инициализируем веса и смещения
        self.randomWeights(listNodeValues)
        self.randomBias(listNodeValues)

        self.view.statusbar["text"] = "Нейросеть создана!"
    
    def saveNetwork(self, filename):
        """Сохраняет веса и смещения в файл."""
        try:
            np.savez(filename, weights=self.listWeight, biases=self.listBias)
            self.view.statusbar["text"] = f"Нейросеть сохранена в {filename}"
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить нейросеть: {e}")
    
    def loadNetwork(self, filename):
        """Загружает веса и смещения из файла."""
        try:
            data = np.load(filename, allow_pickle=True)
            self.listWeight = data['weights'].tolist()
            self.listBias = data['biases'].tolist()
            
            # Обновляем отображение
            self.flattenListWeight(self.listWeight)
            self.view.updateWeight(self.listFlatWeight)
            self.flattenListBias(self.listBias)
            self.view.updateBias(self.listFlatBias)
            
            self.view.statusbar["text"] = f"Нейросеть загружена из {filename}"
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить нейросеть: {e}")


# ==================== ТЕСТОВЫЙ ЗАПУСК ====================
if __name__ == "__main__":
    cfg.tprint("Модуль neural_network загружен")