import numpy as np
import math


from matplotlib import pyplot as plt
from scipy.signal import freqz


def generar_proceso_bernoulli(N, p):
    # Genera la secuencia de bits con probabilidad p de 1 y 1-p de 0
    secuencia = np.random.choice([0, 1], size=N, p=[1-p, p])

    for i in range(len(secuencia)):
        if secuencia[i] == 0:
            secuencia[i] = -1


    return secuencia

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




def LMS(y, d, x,estimacion_inicial):
    N = len(y)
    y = np.array(y)
    x = np.array(x)
    w = estimacion_inicial

    w_estimado = np.zeros((d, N-d))
    x_estimado = np.zeros(N)
    error = np.zeros(N)
    u = 0.9

    for i in range(d, N):
        x_estimado[i] = np.dot(w, y[i-d:i])
        error[i] = x[i] - x_estimado[i]
        w = w + u * y[i-d:i] * error[i]
        w_estimado[:, i-d] = w  # Corrige el índice para llenar w_estimado correctamente

    return w,x_estimado



impulse_response=[1,0.4,0.3,0.1,-0.2,0.05]
#impulse_response=[0.5,1,0.2,0.1,0.05,0.01]
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

N = 1000
p = 0.5  # Probabilidad de obtener un 1

h=[1,0.5]

x=generar_proceso_bernoulli(N,p)
impulse_response=h
#Calculo la respuesta del canal de comuniaciones (en esta caso el h que invente)
y=np.convolve(x,impulse_response)
d=len(impulse_response)
print(y)
estimaciones= np.zeros((d,len(x)-d))

M=500

#Cantidad de iteracion del LMS

estimacion_inicial=np.zeros(len(h))
coeficientes=np.zeros((len(h),M))
coeficientes[:,0]=np.zeros((len(h)))


for i in range(M-1):
    x=generar_proceso_bernoulli(N,p)
    y=np.convolve(x,impulse_response)
    coeficientes[:,i+1], estimacion_salida=LMS(x,d,y,coeficientes[:,i])









# Fase de la respuesta en frecuencia
# Crear un gráfico
plt.figure(figsize=(10, 6))
plt.plot(coeficientes[0,:], marker='o', linestyle='-', color='b')
plt.title('Coeficientes ')
plt.xlabel('Posición')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(estimaciones[1,:], marker='o', linestyle='-', color='b')
plt.title('Valores del Vector por Posición')
plt.xlabel('Posición')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

#funcion_costo=(1/iter)*sum(abs(x_estimacdo-x_real)^2)

