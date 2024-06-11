import numpy as np
from numpy import random
from numpy.random import normal
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import random

from Aux import autocorr_, psd, welch_psd, corr_cruzada


def generar_proceso_bernoulli(N, p):
    # Genera la secuencia de bits con probabilidad p de 1 y 1-p de 0
    secuencia = np.random.choice([0, 1], size=N, p=[1-p, p])

    for i in range(len(secuencia)):
        if secuencia[i] == 0:
            secuencia[i] = -1


    return secuencia





# Define la longitud del proceso Bernoulli y la probabilidad p
N = 200
p = 0.5  # Probabilidad de obtener un 1


x=generar_proceso_bernoulli(N,p)

plt.figure(figsize=(10, 4))
plt.plot(x, marker='o', linestyle='None')  # Gráfica de puntos
plt.title('Proceso de Bernoulli ')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.ylim(-1.5, 1.5)  # Ajuste de los límites del eje y
plt.grid(True)
plt.show()


#Declaro un modelo del canal en tiempo discreto

h=[0.5,1,0.2,0.1,0.05,0.01]
u=0
sigma=0.002
ruido=normal(u,sigma,N)

plt.figure(figsize=(10, 4))
plt.plot(ruido, marker='o', linestyle='None')  # Gráfica de puntos
plt.title('Muestras del ruido ')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()
print(len(h))

print(len(x))

y=np.convolve(x,h)
y = y[:-(len(h)-1)] #Elimino los ultimos elementos para que quede del mismo largo de x

print(len(y))

plt.figure(figsize=(10, 4))
plt.plot(y, marker='o', linestyle='None')  # Gráfica de puntos
plt.title('Respuesta del canal(discreto) a x')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()



Ry = autocorr_(y)
Rx=autocorr_(x)

print(len(x))
psd_y=welch_psd(Ry,30,15)
psd_x=welch_psd(Rx,len(x),25)
print(len(psd_x))



plt.figure(figsize=(10, 4))
plt.plot(Rx, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('Correlacion X')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(psd_x, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('PSD X')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(Ry, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('Correlacion Y')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(psd_y, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('PSD Y')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()



##Ejericio2

Ry=autocorr_(y)
Ryx= corr_cruzada(y,x)






