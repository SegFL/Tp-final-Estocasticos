import numpy as np
import math

from matplotlib import pyplot as plt
from scipy.signal import freqz


def autocorr_(x):
    return corr_cruzada(x,x)

def corr_cruzada(x,y):
    x = np.array(x)  # Asegurarse de que x es un array de NumPy
    y = np.array(y)  # Asegurarse de que x es un array de NumPy

    N = len(x)
    autocorr_values = np.zeros(N)  # Inicializa un array de ceros

    for k in range(N):  # Itera hasta N
        autocorr_values[k] = np.sum(x[k:] * y[:N-k]) / N
    return autocorr_values

def psd(x):
    R_xx = autocorr_(x)  # Calcula la autocorrelación
    PSD = np.fft.fft(R_xx)  # Calcula la transformada de Fourier de la autocorrelación
    return np.abs(PSD)  # La PSD es el valor absoluto de la transformada de Fourier


#M es la longitud de la señal.
#K es el desplazamiento entre segmentos
def welch_psd(x, largo_ventana, paso):
    N = len(x)
    L = (N - largo_ventana) // paso + 1  # Número de segmentos con solapamiento del 50%
    window = np.hamming(largo_ventana)
    V = np.sum(abs(window)**2) / largo_ventana  # Potencia de la ventana
    psd = np.zeros(largo_ventana)

    for i in range(L):
        start = i * paso
        end = start + largo_ventana
        if end > N:
            break
        segment = x[start:end] * window
        segment_fft = np.fft.fft(segment)
        psd += np.abs(segment_fft)**2

    psd = psd / ( largo_ventana * V)  # Normalización

    return psd



def entrenar_filtro_LMS(d):


    N=len(y)

    w= np.zeros(k)
    x_estimado=np.zeros(N)
    error=np.zeros(N)
    u=0.5

    for i in range(k):
        #w[i]=x[k]#Tomo las primeras k muestras de la salida del canal como mi filtro
        x_estimado[i]=np.dot(w,y[i:i+k])

    for i in range(N-k):
        x_estimado[i]=np.dot(w,y[i:i+k])
        error[i]=x[i]-x_estimado[i]
        w[i+1]=w[i]+u*y[i]*error[i]
    return


def LMS(y,d,x):

    N=len(y)
    w= np.zeros(d)

    w_estimado= np.zeros(d)
    x_estimado=np.zeros(N)
    error=np.zeros(N)
    u=0.005

    #El filtro inicial son las primera D-1 muestras de la salida del canal
    #for i in range(d):
       # w[i]=x[d]#Tomo las primeras k muestras de la salida del canal como mi filtro


    for i in range(d,N):
            x_estimado[i]=np.dot(w,y[i:i+d])
            error[i]=x[i]-x_estimado[i]
            w_estimado[i+1]=w_estimado[i]+ u*y[i:i+d]*error[i]
    return




impulse_response=[0.5,1,0.2,0.1,0.05,0.01]
w, h = freqz(impulse_response)


# Magnitud de la respuesta en frecuencia
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.title('Respuesta en Frecuencia')
plt.xlabel('Frecuencia normalizada (rad/muestra)')
plt.ylabel('Magnitud (dB)')
plt.grid()

# Fase de la respuesta en frecuencia
plt.figure()
plt.plot(w, np.angle(h), 'r')
plt.title('Fase de la Respuesta en Frecuencia')
plt.xlabel('Frecuencia normalizada (rad/muestra)')
plt.ylabel('Fase (radianes)')
plt.grid()

plt.show()



