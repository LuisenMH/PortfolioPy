# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:09:13 2023

@author: LuigiMH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
archivo = pd.read_csv("JPM.csv")

#Creación de Columnas
archivo["Rendimientos(Open)"] = 0
archivo["OpenNormalizado"] = 0


#Cálculo de Rendimientos
for i in range (0, len(archivo["Open"])-1):
    archivo["OpenNormalizado"].iloc[i] = archivo["Open"].iloc[i]/max(archivo["Open"]) #IDKIDK

plt.plot(archivo["OpenNormalizado"])
plt.plot(archivo["Rendimientos(Open)"])
plt.ylabel("Rendimientos Vs. Open")
plt.xlabel("Fechas")
plt.title("Comparación")
plt.savefig("OpenRNVsDate")

#Cálculo de números random
randomm = []
mu = (archivo["Rendimientos(Open)"]).iloc[400:800].mean()
sigma = (archivo["Rendimientos(Open)"]).iloc[400:800].std()
for i in range (0,1000):
    randomm.append(random.gauss(mu, sigma))

#Histograma usando random
plt.hist(random)

#Simulación
activo = [archivo["Open"].mean()]
activo = archivo["Open"].iloc[800]

#Valores de Activos
activos=[]
suma=0
for i in range(0, 1000):
    suma = suma + random[i]
    activos.append(activo*math.exp(suma))

#Matriz de Caminos Random
MA = np.zeros(50,1000)
for i in range (0,50):
    for j in range(0,1000):
        MA[i,j] = randomm.gauss(mu,sigma)

#Se puede usar:
VA=np.zeros((50,1000))
for i in range(0,50):
    suma=0
    for j in range(1,1000):
        suma=suma+MA[i,j]
        VA[i,j] = activo*math.exp(suma)