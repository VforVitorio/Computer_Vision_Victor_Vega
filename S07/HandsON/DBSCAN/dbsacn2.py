import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from skimage import io
from skimage.transform import resize
import gc  # Garbage collector

# 1. Cargar y redimensionar la imagen para reducir la memoria necesaria
imagen_original = io.imread('2.png')
print(f"Forma original de la imagen: {imagen_original.shape}")

# Redimensionar a una resolución más baja (ajusta según sea necesario)
max_dimension = 200  # Limitar la dimensión máxima a 200 píxeles
height, width = imagen_original.shape[:2]
scale = max_dimension / max(height, width)
new_height, new_width = int(height * scale), int(width * scale)

imagen = resize(imagen_original, (new_height, new_width), anti_aliasing=True)
print(f"Forma redimensionada: {imagen.shape}")

# Liberar memoria
del imagen_original
gc.collect()

# 2. Preprocesar la imagen
altura, ancho, canales = imagen.shape
pixels = imagen.reshape(altura * ancho, canales)

# 3. Usar submuestreo para reducir más los datos (opcional)
# Submuestrear cada N píxeles para reducir el conjunto de datos
sample_rate = 2  # Tomar cada 2 píxeles
pixels_submuestra = pixels[::sample_rate]
print(f"Píxeles originales: {pixels.shape}, píxeles submuestreados: {pixels_submuestra.shape}")

# Liberar memoria
del pixels
gc.collect()

# 4. Aplicar DBSCAN con parámetros ajustados para usar menos memoria
# - Mayor eps reduce el número de cálculos de vecindad 
# - Mayor min_samples reduce la complejidad de los clusters
dbscan = DBSCAN(
    eps=0.5,               # Valor más alto que el ejemplo anterior
    min_samples=15,        # Valor más alto que el ejemplo anterior
    algorithm='kd_tree',   # 'ball_tree' puede ser una alternativa si kd_tree falla
    leaf_size=50,          # Valor más alto para reducir la complejidad del árbol
    n_jobs=-1              # Usar todos los núcleos disponibles
)

# 5. Ajustar el modelo a los datos submuestreados
clusters = dbscan.fit_predict(pixels_submuestra)

# 6. Reconstruir la imagen segmentada (considerando el submuestreo)
# Crear una matriz de etiquetas del tamaño de la imagen
labels_completa = np.full(altura * ancho, -1)

# Asignar etiquetas a los píxeles que fueron procesados
indices_submuestreados = np.arange(0, altura * ancho, sample_rate)
labels_completa[indices_submuestreados] = clusters

# Rellenar los píxeles no procesados con la etiqueta del píxel más cercano procesado
for i in range(altura * ancho):
    if i not in indices_submuestreados:
        # Encontrar el índice submuestreado más cercano
        idx = (i // sample_rate) * sample_rate
        if idx >= altura * ancho:
            idx = ((i // sample_rate) - 1) * sample_rate
        # Asignar la misma etiqueta
        if idx < altura * ancho:
            labels_completa[i] = labels_completa[idx]

# Reorganizar en la forma de la imagen
imagen_segmentada = labels_completa.reshape(altura, ancho)

# 7. Visualizar resultados
plt.figure(figsize=(12, 6))

# Imagen redimensionada
plt.subplot(121)
plt.imshow(imagen)
plt.title('Imagen Redimensionada')
plt.axis('off')

# Imagen segmentada
plt.subplot(122)
plt.imshow(imagen_segmentada, cmap='nipy_spectral')
plt.title(f'Segmentación DBSCAN (núm. clusters: {len(np.unique(clusters))})')
plt.axis('off')

plt.tight_layout()
plt.savefig('segmentacion_resultado.png')  # Guardar resultado antes de mostrarlo
plt.show()

# 8. Análisis de resultados
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_ruido = list(clusters).count(-1)
print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de puntos de ruido: {n_ruido}")