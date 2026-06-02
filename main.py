#!/usr/bin/env python3
"""
Graphic Neural Network - Главный модуль
Запуск приложения для визуализации нейронных сетей
"""


import sys
import tkinter as tk
import config as cfg

from view import View
from neural_network import NeuralNetwork


def on_closing(mainwin, neural_net=None):
    """Обработчик закрытия окна"""
    if neural_net and hasattr(neural_net, 'cleanup'):
        neural_net.cleanup()
    mainwin.quit()
    mainwin.destroy()
    sys.exit(0)


def main():
    """Запуск приложения"""
    # Создаем главное окно
    mainwin = tk.Tk()
    mainwin.title("Приложение для визуализации нейронных сетей")
    mainwin.geometry(f"{cfg.windowWidth}x{cfg.windowHeight}")
    
    # Создаем компоненты
    brain = NeuralNetwork()      # ядро нейросети (контроллер)
    gui = View(mainwin)          # графический интерфейс (представление)
    
    # Связываем компоненты друг с другом
    brain.setView(gui)
    gui.setController(brain)
    
    # Настраиваем и запускаем интерфейс
    gui.setup()
    
    # Регистрируем обработчик закрытия окна
    mainwin.protocol("WM_DELETE_WINDOW", lambda: on_closing(mainwin, brain))
    
    # Запускаем главный цикл приложения
    mainwin.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)