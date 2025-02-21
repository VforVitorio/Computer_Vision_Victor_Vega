import cv2
import imutils
import numpy as np
import argparse
import random

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-n", "--sections", type=int, required=True,
                help="number of sections (NxN)")
args = vars(ap.parse_args())

# Cargar imagen
image = cv2.imread(args["image"])
if image is None:
    print(f"Error: No se pudo cargar la imagen '{args['image']}'")
    print("Verifique que la ruta sea correcta y que el archivo exista")
    exit(1)

(h, w) = image.shape[:2]

# Calcular dimensiones de secciones
section_h = h // args["sections"]
section_w = w // args["sections"]

# Crear imagen resultado
result = np.zeros((h, w, 3), dtype=np.uint8)

# Procesar cada sección
for i in range(args["sections"]):
    for j in range(args["sections"]):
        # Extraer sección
        start_y = i * section_h
        start_x = j * section_w
        section = image[start_y:start_y +
                        section_h, start_x:start_x + section_w]

        # Rotar aleatoriamente
        angle = random.randint(0, 360)
        rotated = imutils.rotate_bound(section, angle)

        # Redimensionar al tamaño original de la sección
        rotated = cv2.resize(rotated, (section_w, section_h))

        # Colocar en resultado
        result[start_y:start_y + section_h,
               start_x:start_x + section_w] = rotated

# Mostrar resultados
cv2.imshow("Original", image)
cv2.imshow("Collage", result)
cv2.imwrite("collage_result.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# python dynamic_collage.py -i opencv_logo.png -n 3
# Comando correcto
