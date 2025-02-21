from pyniryo import *
import cv2 as cv
import numpy as np
from utlities import LEFT_VISION_AREA

# CONNECTION
IP_ADDRESS = "10.10.10.10"
robot = NiryoRobot(IP_ADDRESS)
robot.calibrate_auto()

# Move to vision area
robot.move_pose(LEFT_VISION_AREA)

# Get image
img_compressed = robot.get_img_compressed()
img = uncompress_image(img_compressed)
img_workspace = extract_img_workspace(img, workspace_ratio=1.0)


def detectar_pieza(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower = np.array([0, 50, 50])
    upper = np.array([180, 255, 255])

    mask = cv.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    object_positions = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 500:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                object_positions.append((cx, cy))
                # Draw center point
                cv.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    return img, object_positions


def relative_pos_from_pixels(img, px, py):
    """
    Convierte coordenadas de píxeles (px, py) a coordenadas relativas
    del área de trabajo del robot. Esta conversión depende de la
    calibración y se asume un mapeo lineal.
    """
    height, width = img.shape[:2]
    # Establece los límites calibrados del área de trabajo (en metros, por ejemplo)
    x_min, x_max = -0.2, 0.2    # Límites en eje X
    y_min, y_max = -0.2, 0.2    # Límites en eje Y

    # Conversión lineal
    relative_x = x_min + (px / width) * (x_max - x_min)
    relative_y = y_min + (py / height) * (y_max - y_min)
    return relative_x, relative_y


def get_target_pose_from_rel(x_rel, y_rel, z=0.1, roll=0.0, pitch=0.0, yaw=0.0):
    """
    Crea y retorna un PoseObject a partir de coordenadas relativas.
    Ajusta z y la orientación según tu configuración.
    """
    return PoseObject(x_rel, y_rel, z, roll, pitch, yaw)


result_img_workspace, positions = detectar_pieza(img_workspace)
cv.imshow("Detected Objects", result_img_workspace)
cv.waitKey(0)
cv.destroyAllWindows()

print("Detected object positions:", positions)


for (px, py) in positions:
    x_rel, y_rel = relative_pos_from_pixels(img_workspace, px, py)
    print(
        f"Moviendo a las coordenadas relativas para el pixel ({px}, {py}): ({x_rel:.3f}, {y_rel:.3f})")
    target_pose = get_target_pose_from_rel(x_rel, y_rel)
    robot.move_pose(target_pose)
    # Puedes agregar un comando para agarrar la pieza aquí, por ejemplo:
    # robot.grasp()


robot.move_pose(-0.0034, -0.0738, 0.2515, -2.330, 1.436, 1.886)

robot.go_to_sleep()
