# Lenguaje: Python
# filepath: S04/Wavelets/sinusoidal.py

import numpy as np
import matplotlib.pyplot as plt


def exercise2_frequency_analysis():
    """
    Exercise 2 - Frequency Analysis

    Generación de señales sinusoidales con diferentes frecuencias,
    combinación de señales y visualización de sus transformadas.
    Muestra cómo se representan diferentes frecuencias en el espectro.
    """
    N = 200
    x = np.linspace(0, 2*np.pi, N)
    y = np.linspace(0, 2*np.pi, N)
    X, Y = np.meshgrid(x, y)

    # Señal 1: Coseno que varía a lo largo del eje X
    f1 = 5  # frecuencia para la señal 1
    signal1 = np.cos(f1 * X)

    # Señal 2: Seno que varía a lo largo del eje Y
    f2 = 8  # frecuencia para la señal 2
    signal2 = np.sin(f2 * Y)

    # Combinación de las dos señales
    combined_signal = signal1 + signal2

    # Cálculo de las transformadas de Fourier
    ft_signal1 = np.fft.fftshift(np.fft.fft2(signal1))
    ft_signal2 = np.fft.fftshift(np.fft.fft2(signal2))
    ft_combined = np.fft.fftshift(np.fft.fft2(combined_signal))

    # Visualización de las señales y sus transformadas
    plt.figure(figsize=(12, 8))

    # Señal 1 y su FFT
    plt.subplot(331)
    plt.imshow(signal1, cmap='gray')
    plt.title('Señal 1: Coseno (X)')
    plt.colorbar()

    plt.subplot(332)
    plt.imshow(np.log(1 + np.abs(ft_signal1)), cmap='gray')
    plt.title('FFT Señal 1')
    plt.colorbar()

    # Señal 2 y su FFT
    plt.subplot(334)
    plt.imshow(signal2, cmap='gray')
    plt.title('Señal 2: Seno (Y)')
    plt.colorbar()

    plt.subplot(335)
    plt.imshow(np.log(1 + np.abs(ft_signal2)), cmap='gray')
    plt.title('FFT Señal 2')
    plt.colorbar()

    # Señal combinada y su FFT
    plt.subplot(337)
    plt.imshow(combined_signal, cmap='gray')
    plt.title('Señal Combinada')
    plt.colorbar()

    plt.subplot(338)
    plt.imshow(np.log(1 + np.abs(ft_combined)), cmap='gray')
    plt.title('FFT Señal Combinada')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


exercise2_frequency_analysis()
