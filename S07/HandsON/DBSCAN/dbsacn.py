import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import io
from sklearn.preprocessing import StandardScaler

# 1. Cargar la imagen
imagen = io.imread('2.png')
print(f"Forma de la imagen: {imagen.shape}")

# 2. Preprocesar la imagen
# Redimensionar si es necesario para imágenes grandes
altura, ancho, canales = imagen.shape
pixels = imagen.reshape(altura * ancho, canales)

# 3. Escalar los datos para mejorar el rendimiento de DBSCAN
pixels_escalados = StandardScaler().fit_transform(pixels)

# 4. Aplicar DBSCAN
# Parámetros clave:
# - eps: distancia máxima entre dos muestras para ser consideradas en el mismo cluster
# - min_samples: número mínimo de muestras en un vecindario para que un punto sea considerado core point
dbscan = DBSCAN(eps=0.3, min_samples=10)
clusters = dbscan.fit_predict(pixels_escalados)

# 5. Crear imagen segmentada
# Reorganizar las etiquetas del clúster a la forma de la imagen
imagen_segmentada = clusters.reshape(altura, ancho)

# 6. Visualizar resultados
plt.figure(figsize=(15, 8))

# Imagen original
plt.subplot(121)
plt.imshow(imagen)
plt.title('Imagen Original')
plt.axis('off')

# Imagen segmentada
plt.subplot(122)
plt.imshow(imagen_segmentada, cmap='nipy_spectral')
plt.title(f'Segmentación DBSCAN (núm. clusters: {len(np.unique(clusters))})')
plt.axis('off')

plt.tight_layout()
plt.show()

# 7. Análisis de resultados
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_ruido = list(clusters).count(-1)
print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de puntos de ruido: {n_ruido}")

# 8. Opcional: Crear imagen donde cada segmento tenga el color promedio de ese segmento
resultado = np.zeros_like(imagen)

for etiqueta in np.unique(clusters):
    if etiqueta == -1:
        # Color negro para ruido
        color = [0, 0, 0]
    else:
        # Color promedio del clúster
        mascara = clusters == etiqueta
        color = np.mean(pixels[mascara], axis=0).astype(int)
    
    # Aplicar color a todos los píxeles del clúster
    resultado.reshape(-1, canales)[clusters == etiqueta] = color

# Mostrar imagen con colores promedio
plt.figure(figsize=(8, 8))
plt.imshow(resultado.reshape(altura, ancho, canales))
plt.title('Segmentos con colores promedio')
plt.axis('off')
plt.show()