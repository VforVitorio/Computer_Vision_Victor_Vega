import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from matplotlib import pyplot as plt

# Cargar la imagen
imagen = cv2.imread('ejeomplo.jpg')
# Convertir de BGR a RGB (para mostrar correctamente con matplotlib)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Redimensionar la imagen para acelerar el procesamiento (opcional)
escala = 0.5
imagen_redim = cv2.resize(imagen_rgb, (0, 0), fx=escala, fy=escala)

# Aplanar la imagen en una matriz 2D (píxeles x características)
flat_imagen = imagen_redim.reshape((-1, 3))
flat_imagen = np.float32(flat_imagen)

# Estimar el ancho de banda (parámetro importante para Mean Shift)
bandwidth = estimate_bandwidth(flat_imagen, quantile=0.1, n_samples=50)

# Aplicar el algoritmo Mean Shift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(flat_imagen)

# Obtener las etiquetas de los segmentos y los centros
etiquetas = ms.labels_
centros = ms.cluster_centers_
n_clusters = len(np.unique(etiquetas))

# Crear la imagen segmentada reemplazando cada píxel por su centro correspondiente
segmentada = centros[etiquetas].reshape(imagen_redim.shape)
segmentada = np.uint8(segmentada)

# Mostrar los resultados
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(imagen_redim)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(122)
plt.imshow(segmentada)
plt.title(f'Imagen Segmentada: {n_clusters} segmentos')
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Número de segmentos encontrados: {n_clusters}")