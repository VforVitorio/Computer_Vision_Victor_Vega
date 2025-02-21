import numpy as np
import cv2
import glob


def calibrar_camara(ruta_imagenes, patron_size=(9, 6)):
    """
    Función para calibrar una cámara usando imágenes de un tablero de ajedrez

    Parámetros:
    ruta_imagenes: String con la ruta a las imágenes (ej: './calibracion/*.jpeg')
    patron_size: Tupla con el número de esquinas internas del tablero (ancho, alto)

    Retorna:
    ret: Bool indicando si la calibración fue exitosa
    mtx: Matriz de la cámara
    dist: Coeficientes de distorsión
    rvecs: Vectores de rotación
    tvecs: Vectores de traslación
    """

    # Preparar puntos objeto 3D
    objp = np.zeros((patron_size[0] * patron_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:patron_size[0], 0:patron_size[1]].T.reshape(-1, 2)

    # Arrays para almacenar puntos objeto y puntos imagen
    objpoints = []  # Puntos 3D en espacio real
    imgpoints = []  # Puntos 2D en el plano de la imagen

    # Obtener lista de imágenes
    imagenes = glob.glob(ruta_imagenes)

    for fname in imagenes:
        # Leer imagen
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontrar esquinas del tablero
        ret, corners = cv2.findChessboardCorners(gray, patron_size, None)

        if ret:
            # Refinar las esquinas encontradas
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Agregar puntos
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Dibujar y mostrar las esquinas
            cv2.drawChessboardCorners(img, patron_size, corners2, ret)
            cv2.imshow('Esquinas detectadas', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrar cámara
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return ret, mtx, dist, rvecs, tvecs


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta a las imágenes de calibración
    ruta_imagenes = '*.jpeg'
    # Calibrar cámara
    ret, mtx, dist, rvecs, tvecs = calibrar_camara(ruta_imagenes)

    if ret:
        print("Calibración exitosa!")
        print("\nMatriz de la cámara:")
        print(mtx)
        print("\nCoeficientes de distorsión:")
        print(dist)
    else:
        print("La calibración falló")

# Para usar los parámetros de calibración en una imagen:


def undistort_imagen(imagen, mtx, dist):
    """
    Corrige la distorsión en una imagen usando los parámetros de calibración
    """
    return cv2.undistort(imagen, mtx, dist, None, mtx)
