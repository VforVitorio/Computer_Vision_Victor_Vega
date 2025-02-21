from pyniryo import *

# CONSTANS
IP_ADDRESS = "10.10.10.10"
GREEN_MIN_HSV = [40, 120, 75]
GREEN_MAX_HSV = [90, 255, 255]
ANY_MIN_HSV = [0, 50, 100]
ANY_MAX_HSV = [179, 255, 255]

robot = NiryoRobot(IP_ADDRESS)
robot.calibrate_auto()

observation_pose = PoseObject(
    -0.0034, -0.1878, 0.2515,
    -2.330, 1.436, 1.886
)
robot.move_pose(observation_pose)

img_compressed = robot.get_img_compressed()
img = uncompress_image(img_compressed)

color_input = input("Enter color (RED, BLUE, GREEN, etc.): ").upper()

if color_input == "GREEN":
    # Usa los límites HSV de 'GREEN_MIN_HSV' y 'GREEN_MAX_HSV'
    img_threshold = threshold_hsv(img, GREEN_MIN_HSV, GREEN_MAX_HSV)
else:
    # De lo contrario, usa la clase enum de pyniryo
    color_enum = ColorHSV[color_input]
    img_threshold = threshold_hsv(img, *color_enum.value)

img_close = morphological_transformations(
    img_threshold,
    morpho_type=MorphoType.CLOSE,
    kernel_shape=(11, 11),
    kernel_type=KernelType.ELLIPSE
)

img_erode = morphological_transformations(
    img_threshold,
    morpho_type=MorphoType.ERODE,
    kernel_shape=(9, 9),
    kernel_type=KernelType.RECT
)

show_img("img_threshold", img_threshold)
show_img("img_close", img_close)
show_img_and_wait_close("img_erode", img_erode)
