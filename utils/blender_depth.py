import bpy
import json
import numpy as np
import os
import time
from PIL import Image


def add_lighting():
    """Add uniform lighting to ensure no shadows and even illumination"""
    # Clear existing lights in the scene
    bpy.ops.object.select_by_type(type='LIGHT')
    bpy.ops.object.delete()

    # Create ambient light through world background
    bpy.context.scene.world.use_nodes = True
    world_node_tree = bpy.context.scene.world.node_tree
    bg_node = world_node_tree.nodes.get("Background")
    bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # White color for ambient light
    bg_node.inputs[1].default_value = 0.5  # Strength for even illumination

    # Add point lights for coverage
    positions = [
        (3.4154, 4.6753, 6.5981),
        (0.80797, -7.77557, 4.78247),
        (-4.96121, 1.9155, 9.01307),
        (-0.76353, -4.182, 1.5501)
    ]
    for pos in positions:
        bpy.ops.object.light_add(type="POINT", location=pos)
        light = bpy.context.object
        light.data.energy = 200
        light.data.use_shadow = False  # Disable shadows


def calculate_distance(camera_name, object_name):
    """Calculate the distance between the camera and object"""
    cam = bpy.data.objects.get(camera_name)
    obj = bpy.data.objects.get(object_name)

    if not cam or not obj:
        print(f"Camera or object not found: {camera_name}, {object_name}")
        return None

    cam_location = cam.matrix_world.translation
    obj_location = obj.matrix_world.translation
    return (cam_location - obj_location).length


def adjust_camera_clipping(camera_name, object_name):
    """Dynamically adjust camera clipping based on object distance and size"""
    cam = bpy.data.objects[camera_name]
    obj = bpy.data.objects[object_name]

    # Estimate object extent
    bbox_min = np.min([v[:] for v in obj.bound_box], axis=0)
    bbox_max = np.max([v[:] for v in obj.bound_box], axis=0)
    obj_extent = max(bbox_max - bbox_min)

    # Calculate distance to the object
    distance = calculate_distance(camera_name, object_name)

    # Set clipping range
    cam.data.clip_start = max(0.01, distance - obj_extent * 1.5)
    cam.data.clip_end = distance + obj_extent * 1.5
    print(f"Camera clipping set: start={cam.data.clip_start}, end={cam.data.clip_end}")


def setup_render_layers(camera_name, object_name):
    """ Set up the render layers to include masks and normalized depth """
    view_layer = bpy.context.view_layer
    view_layer.use_pass_combined = True
    view_layer.use_pass_z = True  # Depth pass
    view_layer.use_pass_object_index = True  # Object masks
    
      # Set object indices for the masks
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.pass_index = 1  # Assign index for mask pass

    distance = calculate_distance(camera_name, object_name)
    from_min = 0
    from_max = distance + 5.0  # Adjust dynamically based on object distance

    # Enable compositor and clear previous nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Render layers
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    # Depth normalization
    depth_map_range = tree.nodes.new(type="CompositorNodeMapRange")
    depth_map_range.inputs['From Min'].default_value = 0
    depth_map_range.inputs['From Max'].default_value = 70
    depth_map_range.inputs['To Min'].default_value = 0
    depth_map_range.inputs['To Max'].default_value = 1
    links.new(render_layers.outputs['Depth'], depth_map_range.inputs[0])

    # Depth output
    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.base_path = f"{ROOT}/{stage}/depth/"
    depth_file_output.format.file_format = 'PNG'
    depth_file_output.format.color_depth = '16'
    depth_file_output.format.color_mode = 'BW'
    links.new(depth_map_range.outputs[0], depth_file_output.inputs[0])

    # # Mask output
    # mask_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # mask_file_output.label = 'Mask Output'
    # mask_file_output.base_path = f"{ROOT}/{stage}/mask/"
    # mask_file_output.format.file_format = 'PNG'
    # mask_file_output.format.color_mode = 'BW'
    # links.new(render_layers.outputs['IndexOB'], mask_file_output.inputs[0])
    
     # Create mask output node (object index)
    mask_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    mask_file_output.label = 'Mask Output'
    mask_file_output.base_path = f"{ROOT}/{stage}/mask/"
    mask_file_output.format.file_format = 'PNG'
    mask_file_output.format.color_mode = 'BW'
    links.new(render_layers.outputs['IndexOB'], mask_file_output.inputs[0])

    # Add node to combine depth output and mask
    mask_multiply = tree.nodes.new(type="CompositorNodeMath")
    mask_multiply.operation = 'MULTIPLY'
    links.new(render_layers.outputs['IndexOB'], mask_multiply.inputs[0])
    links.new(depth_map_range.outputs[0], mask_multiply.inputs[1])

    # Output final depth with mask applied
    final_output = tree.nodes.new(type="CompositorNodeOutputFile")
    final_output.label = 'Final Depth Output'
    final_output.base_path = f"{ROOT}/{stage}/final_depth/"
    final_output.format.file_format = 'PNG'
    final_output.format.color_depth = '16'
    final_output.format.color_mode = 'BW'
    final_output.format.compression = 0  # Grayscale depth
    links.new(mask_multiply.outputs[0], final_output.inputs[0])



def camera_setting(exp_dir, n_frames=100, stage='train', track_to='start'):
    '''Set the render setting for the camera and the scene'''
    bpy.context.scene.render.engine = 'CYCLES'  # Use Cycles for more control (change to your preferred engine) or BLENDER_WORKBENCH for normal images
    bpy.context.scene.cycles.samples = 64  # Set samples for Cycles

    add_lighting()

    cam_obj = bpy.data.objects['Camera']
    print(cam_obj.name)
    
      # Calculate distance to object for clipping range
    distance = calculate_distance("Camera", track_to)
    # clip_start = 0
    # clip_end = 100  # Default to 100 if distance fails
    
    # # Set the clipping distance for the camera
    # cam_obj.data.clip_start = clip_start
    # cam_obj.data.clip_end = clip_end
    # # Set the camera type to orthographic
    # # cam_obj.data.type = 'ORTHO'
    # bpy.data.cameras["Camera"].clip_start= clip_start
    # bpy.data.cameras["Camera"].clip_end = clip_end
    # print(f"Camera clipping set: start = {clip_start}, end = {clip_end}")
    
    # bpy.context.scene.view_settings.exposure = -1.0  # Lower exposure if the scene is too bright
    constraint = cam_obj.constraints.new(type='TRACK_TO')
    constraint.target = bpy.data.objects[track_to]
    bpy.context.scene.frame_end = n_frames - 1
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_step = 1
    bpy.context.scene.render.fps = 1  # Reduces the frames per second (e.g., from 24 or 30 to 10)
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 800
    bpy.context.scene.render.image_settings.color_mode = 'RGB'  # Ensure RGBA
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.filepath = f'{exp_dir}/{stage}/'

    radius = 3.2  # Distance from the origin
    if stage == 'test':  # Turntable
        phi = [np.pi / 3. for i in range(n_frames)]  # Constant pitch for higher view
        theta = [i * (2 * np.pi) / n_frames for i in range(n_frames)]
    else:  # Randomly sampled on the upper hemisphere
        # Avoid directly overhead viewpoints by restricting phi
        phi_min = 0.26  # About 15 degrees from vertical
        phi_max = 0.5 * np.pi  # 90 degrees from vertical

        # Generate random phi and theta within the restricted range
        phi = phi_min + (phi_max - phi_min) * np.random.random_sample(n_frames)
        # phi = np.random.random_sample(n_frames) * 0.5 * np.pi
        theta = np.random.random_sample(n_frames) * 2. * np.pi

    for f in range(n_frames):
        x = radius * np.cos(theta[f]) * np.sin(phi[f])
        y = radius * np.sin(theta[f]) * np.sin(phi[f])
        z = radius * np.cos(phi[f])
        cam_obj.location = np.array([x, y, z])
        cam_obj.keyframe_insert(data_path="location", frame=f)
    

def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

def get_K(camd):
    '''Calculate intrinsic matrix K from the Blender camera data.'''
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = [[s_u, skew, u_0],
        [0., s_v, v_0],
        [0., 0., 1.]]
    return K
        
def get_camera_dict():
    scene = bpy.context.scene
    frame_start = scene.frame_start
    frame_end = scene.frame_end
    
    cam_dict = {}
    cameras = []

    for obj in scene.objects:
        if obj.type != 'CAMERA':
            continue
        cameras.append((obj, obj.data))
    
    for obj, obj_data in cameras:
        K = get_K(obj_data)
        cam_dict['K'] = K

    frame_range = range(frame_start, frame_end + 1)
    for f in frame_range:
        scene.frame_set(f)
        frame_name = '{:0>4d}'.format(f)
        for obj, obj_data in cameras:
            mat = obj.matrix_world
            mat_list =[
                [mat[0][0], mat[0][1], mat[0][2], mat[0][3]],
                [mat[1][0], mat[1][1], mat[1][2], mat[1][3]],
                [mat[2][0], mat[2][1], mat[2][2], mat[2][3]],
                [mat[3][0], mat[3][1], mat[3][2], mat[3][3]],
            ]
            cam_dict[frame_name] = mat_list
    
    return cam_dict


def camera_export(stage='train'):
    file_path = f'{ROOT}/camera_{stage}.json'
    with open(file_path, 'w') as fh:
        cam = get_camera_dict()
        json.dump(cam, fh)


if __name__ == "__main__":
    obj_id = '10905'
    obj_state = 'start'
    obj_category = 'fridge'
    ROOT = f'/media/qil/DATA/Carter_Articulated_Objects/paris-reconstruction/load/sapien/{obj_category}/{obj_id}/{obj_state}'
    n_frames = 100
    stage = 'test'
    track_to = obj_state

    os.makedirs(ROOT, exist_ok=True)
    os.makedirs(f'{ROOT}/{stage}/depth', exist_ok=True)
    os.makedirs(f'{ROOT}/{stage}/mask', exist_ok=True)

    # Remove default cube
    if "Cube" in bpy.data.objects:
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    # Load the object model
    bpy.ops.import_scene.obj(filepath=f"/media/qil/DATA/Carter_Articulated_Objects/paris-reconstruction/data/sapien/{obj_category}/{obj_id}/textured_objs/{obj_state}/{obj_state}.obj")
    loaded_obj = bpy.context.selected_objects[0]

    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1
    bpy.context.scene.render.image_settings.compression = 0

    camera_setting(exp_dir=ROOT, n_frames=n_frames, stage=stage, track_to=track_to)

    # Set up render layers
    setup_render_layers(camera_name="Camera", object_name=obj_state)

    # Export camera parameters
    camera_export(stage)

    # Render images
    bpy.ops.render.render(animation=True)
