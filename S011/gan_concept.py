import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity

# Parte 1: Generar muestras de una distribución (simulando un conjunto de datos de imágenes)
# En computer vision, esto podría representar características extraídas de imágenes
print("Generando datos originales (simulando características de imágenes)...")
n_samples = 1000
centers = [[0, 0], [1, 5], [5, 1]]
X, _ = make_blobs(n_samples=n_samples, centers=centers, random_state=42, cluster_std=0.8)

# Visualizar los datos originales
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Muestras Originales\n(Características de Imágenes)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

# Parte 2: Entrenar un modelo generativo (observar muchas muestras)
# Usamos un modelo mixto gaussiano como ejemplo simple de modelo generativo
print("Entrenando el modelo generativo (GMM)...")
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)
print(f"Modelo GMM entrenado con {gmm.n_components} componentes gaussianas")

# Visualizar el modelo GMM aprendido
x = np.linspace(-2, 7, 100)
y = np.linspace(-2, 7, 100)
X_grid, Y_grid = np.meshgrid(x, y)
XX = np.array([X_grid.ravel(), Y_grid.ravel()]).T
Z = -gmm.score_samples(XX)
Z = Z.reshape(X_grid.shape)

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.contour(X_grid, Y_grid, Z, levels=10, cmap='viridis')
plt.title("Modelo GMM Aprendido\n(Distribución Estimada)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

# Parte 3: Generar nuevas muestras a partir del modelo GMM
print("Generando nuevas muestras a partir del modelo GMM...")
X_new, _ = gmm.sample(n_samples=1000)

# Visualizar las nuevas muestras generadas por GMM
plt.subplot(1, 3, 3)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.5, color='red')
plt.title("Nuevas Muestras Generadas (GMM)\n(Imágenes Sintéticas)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

plt.tight_layout()
plt.show()

# Nuevo ejemplo: Kernel Density Estimation (KDE)
print("\nEntrenando un segundo modelo generativo (KDE)...")
kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde.fit(X)

# Generar nuevas muestras usando KDE
print("Generando nuevas muestras a partir del modelo KDE...")
X_new_kde = kde.sample(n_samples=1000)

# Visualizar los resultados de KDE
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title("Datos Originales")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

plt.subplot(1, 3, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
Z = np.exp(kde.score_samples(XX))
Z = Z.reshape(X_grid.shape)
plt.contour(X_grid, Y_grid, Z, levels=10, cmap='viridis')
plt.title("Modelo KDE Aprendido\n(Distribución Estimada)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

plt.subplot(1, 3, 3)
plt.scatter(X_new_kde[:, 0], X_new_kde[:, 1], alpha=0.5, color='green')
plt.title("Nuevas Muestras Generadas (KDE)\n(Imágenes Sintéticas)")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")

plt.tight_layout()
plt.show()

print("\n¿Qué demuestra este ejemplo?")
print("1. Observamos muestras de una distribución (datos originales)")
print("2. Aprendemos dos modelos diferentes de esta distribución (GMM y KDE)")
print("3. Generamos nuevas muestras 'sintéticas' de la misma distribución")
print("\nEn Computer Vision, este concepto se aplica para:")
print("- Generar nuevas imágenes realistas después de aprender patrones de imágenes existentes")
print("- Aumentar conjuntos de datos para entrenar redes neuronales más robustas")
print("- Crear variaciones de imágenes manteniendo características esenciales")