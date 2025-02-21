import cv2
import imutils
import numpy as np
import argparse


# 1: Cargar una imagen
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")

# 2: Permite seleccionar al usuario un color (rojo, verde, azul) como
# parámetro por línea de comandos
ap.add_argument("-c", "--color", required=True,
                choices=['rojo', 'verde', 'azul'],
                help="color to filter (rojo/verde/azul)")

args = vars(ap.parse_args())


# dimensions, including width, height, and number of channels
image = cv2.imread(args["image"])
(h, w, c) = image.shape[:3]

# Mantenga solo el canal de color seleccionado, poniendo los otros 2 a cero

# Crear copia de la imagen
result = image.copy()

# Definir índices de canales BGR
color_indices = {
    'azul': 0,
    'verde': 1,
    'rojo': 2
}

# Obtener el índice del color seleccionado
selected_index = color_indices[args["color"]]

# Poner a cero los otros canales
for i in range(3):
    if i != selected_index:
        result[:, :, i] = 0

# Mostrar imágenes
cv2.imshow("Imagen Original", image)
cv2.imshow("Imagen Filtrada", result)

# Guardar imagen resultante
output_filename = f"filtered_{args['color']}.jpg"
cv2.imwrite(output_filename, result)

# Esperar tecla y cerrar ventanas
cv2.waitKey(0)
cv2.destroyAllWindows()


# python selective_color_filter.py -i opencv_logo.png -c rojo
