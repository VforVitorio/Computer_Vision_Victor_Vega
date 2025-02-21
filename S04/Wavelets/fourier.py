import numpy as np
import matplotlib.pyplot as plt
import cv2


def exercise1_basic_patterns():
    """
    Exercise 1: Basic Pattern Visualization and their Transforms

    Objective: Create simple patterns (vertical and horizontal lines) and visualize
    their Fourier transforms to understand the relationship between spatial and
    frequency domains.
    """
    # Create image with a single frequency vertical cosine pattern
    x = np.linspace(0, 2*np.pi, 200)
    y = np.linspace(0, 2*np.pi, 200)
    X, Y = np.meshgrid(x, y)
    vertical_lines = np.cos(5*X)  # 5 is the frequency

    # Create image with a single frequency horizontal cosine pattern
    horizontal_lines = np.cos(5*Y)  # 5 is the frequency

    # Calculate Fourier transforms
    ft_vertical = np.fft.fftshift(np.fft.fft2(vertical_lines))
    ft_horizontal = np.fft.fftshift(np.fft.fft2(horizontal_lines))

    # Visualize results
    plt.figure(figsize=(12, 4))

    plt.subplot(231), plt.imshow(vertical_lines, cmap='gray')
    plt.title('Vertical Lines')
    plt.subplot(232), plt.imshow(np.log(1 + np.abs(ft_vertical)), cmap='gray')
    plt.title('FFT Vertical Lines')

    plt.subplot(234), plt.imshow(horizontal_lines, cmap='gray')
    plt.title('Horizontal Lines')
    plt.subplot(235), plt.imshow(
        np.log(1 + np.abs(ft_horizontal)), cmap='gray')
    plt.title('FFT Horizontal Lines')

    plt.tight_layout()
    plt.show()


exercise1_basic_patterns()
