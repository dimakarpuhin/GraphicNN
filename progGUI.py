import tkinter as tk
from tkinter import ttk
from tkinter import *

import pUtility as pu





#Тестовое окно просмотра
if __name__ == "__main__": 
    mainwin = Tk()
    WIDTH = pu.windowWidth
    HEIGHT = pu.windowHeight
    mainwin.geometry("%sx%s" % (WIDTH, HEIGHT))
    mainwin.title("Тестовое окно")

    
    mainwin.mainloop() # запустить главный цикл, чтобы окно могло реагировать на дейтвия пользователя