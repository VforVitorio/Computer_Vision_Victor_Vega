from pyniryo import *

# CONNECTION
IP_ADDRESS = "10.10.10.10"
pick_position = PoseObject(
    -0.0034, -0.1878, 0.2515,
    -2.330, 1.436, 1.886
)

up_position = PoseObject(
    -0.0035, -0.1878, 0.3016,
    -2.323, 1.437, 1.887
)
mid_position = PoseObject(
    0.1806, -0.0034, 0.3,
    -1.822,  1.485, -3.016
)
down_position = PoseObject(
    0.1806, -0.0034, 0.15,
    -1.822,  1.485, -3.016
)


def get_image():
    img_compressed = robot.get_img_compressed()
    img = uncompress_image(img_compressed)
    return extract_img_workspace(img)


def get_object(
        workspace_name: str,
        position_vision: PoseObject,
        form: ObjectShape,
        color: ObjectColor,
        offset_size: float,
        max_catch_count: int
):
    """
    Detects and catches a specific piece and color
    """
    catch_count = 0
    while catch_count < max_catch_count:

        obj_found, shape, color_object = robot.vision_pick(
            workspace_name,
            shape=form,
            color=color
        )
        robot.move_pose(up_position)
        robot.move_pose(mid_position)
        robot.move_pose(down_position)
        robot.release_with_tool()

        if not obj_found:
            robot.wait(0.1)
            next_place_pose = position_vision.copy_with_offsets(
                z_offset=catch_count * offset_size
            )
            robot.place_from_pose(next_place_pose)
            catch_count += 1
            continue


robot = NiryoRobot(IP_ADDRESS)
robot.calibrate_auto()

# Move to vision area
robot.move_pose(pick_position)

# Captura la imagen para establecer el área de visión
get_image()

# Solicita la forma y el color por terminal
shape_input = input("Enter shape (CIRCLE, SQUARE, TRIANGLE, etc.): ")
color_input = input("Enter color (RED, BLUE, GREEN, etc.): ")

# Convierte las cadenas a los enums correspondientes
shape_enum = ObjectShape[shape_input.upper()]
color_enum = ObjectColor[color_input.upper()]

# Llama a la función get_object con la forma y el color introducidos
get_object(
    'right_vision',
    pick_position,
    shape_enum,
    color_enum,
    0.5,
    3
)
