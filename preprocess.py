import pymeshlab
import numpy as np
import json
import os
import open3d as o3d
from utils.axis import save_axis_mesh


def normalize(v):
    return v / np.sqrt(np.sum(v**2))

def rewrite_json_from_urdf(src_root):
    root = 'data/partnet-mobility'
    urdf_file = os.path.join(src_root, 'mobility.urdf')
    from lxml import etree as ET
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    visuals_dict = {}
    for link in root.iter('link'):
        print(link)
        meshes = []
        for visuals in link.iter('visual'):
            meshes.append(visuals[1][0].attrib['filename'])
        visuals_dict.update({link.attrib['name']: meshes})
    
    # load .json file as a dict
    with open(os.path.join(src_root, 'mobility_v2.json'), 'r') as f:
        meta = json.load(f)
        f.close()
    
    # find mesh files in urdf and add to meta
    for entry in meta:
        print(entry.keys())
        link_name = 'link_{}'.format(entry['id'])
        entry['visuals'] = visuals_dict[link_name]
    
    # write a self-used json file
    with open(os.path.join(src_root, 'mobility_v2_self.json'), 'w') as json_out_file:
        json.dump(meta, json_out_file)
        json_out_file.close()

def get_rotation_axis_angle(k, theta):
    '''
    Rodrigues' rotation formula
    args:
    * k: direction vector of the axis to rotate about
    * theta: the (radian) angle to rotate with
    return:
    * 3x3 rotation matrix
    '''
    k = normalize(k)
    kx, ky, kz = k[0], k[1], k[2]
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.zeros((3, 3))
    R[0, 0] = cos + (kx**2) * (1 - cos)
    R[0, 1] = kx * ky * (1 - cos) - kz * sin
    R[0, 2] = kx * kz * (1 - cos) + ky * sin
    R[1, 0] = kx * ky * (1 - cos) + kz * sin
    R[1, 1] = cos + (ky**2) * (1 - cos)
    R[1, 2] = ky * kz * (1 - cos) - kx * sin
    R[2, 0] = kx * kz * (1 - cos) - ky * sin
    R[2, 1] = ky * kz * (1 - cos) + kx * sin
    R[2, 2] = cos + (kz**2) * (1 - cos)
    return R

def merge_meshsets(mss: list):
    for ms in mss:
        ms.generate_by_merging_visible_meshes(mergevisible=True,
                                              deletelayer=False,
                                              mergevertices=True,
                                              alsounreferenced=True)
    return mss

def z_up_frame_meshsets(mss: list):
    for ms in mss:
        ms.compute_matrix_from_rotation(rotaxis='X axis',
                                        rotcenter='origin',
                                        angle=90,
                                        snapflag=False,
                                        freeze=True,
                                        alllayers=True)
    return mss

def save_meshsets_ply(mss: list, fnames: list):
    for ms, fname in zip(mss, fnames):
        ms.save_current_mesh(fname,
                             save_vertex_quality=False,
                             save_vertex_radius=False,
                             save_vertex_color=False,
                             save_face_color=False,
                             save_face_quality=False,
                             save_wedge_color=False,
                             save_wedge_texcoord=False,
                             save_wedge_normal=False)
        # resave with open3d, because there is incompatibility of pymesh with load_ply in pytorch3d for later evaluation
        mesh = o3d.io.read_triangle_mesh(fname)
        o3d.io.write_triangle_mesh(fname, mesh, write_triangle_uvs=False)

def get_arti_info(entry, motion):
    res = {
        'axis': {
            'o': entry['jointData']['axis']['origin'],
            'd': entry['jointData']['axis']['direction']
        }
    }

    # hinge joint
    if entry['joint'] == 'hinge':
        assert motion['type'] == 'rotate'
        R_limit_l, R_limit_r = motion['rotate'][0], motion['rotate'][1]
        res.update({
            'rotate': {
                'l': R_limit_l,  # start state
                'r': R_limit_r  # end state
            },
        })
    # slider joint
    elif entry['joint'] == 'slider':
        assert motion['type'] == 'translate'
        T_limit_l, T_limit_r = motion['translate'][0], motion['translate'][1]
        res.update({
                'translate': {
                'l': T_limit_l,
                'r': T_limit_r
            }
        })
    # other joint
    else:
        raise NotImplemented(
            '{} joint is not implemented'.format(entry['joint']))

    return res

# def load_articulation(src_root, joint_id):
#     with open(os.path.join(src_root, 'mobility_v2_self.json'), 'r') as f:
#         meta = json.load(f)
#         f.close()

#     for entry in meta:
#         if entry['id'] == joint_id:
#             arti_info = get_arti_info(entry, motions['motion']) 

#     return arti_info, meta

def load_articulation(src_root):
    with open(os.path.join(src_root, 'mobility_v2_self.json'), 'r') as f:
        meta = json.load(f)
        f.close()

    articulations = []  # new list to hold all articulation data
    motion = motions['motion']
    R_limit_l, R_limit_r = motion['rotate'][0], motion['rotate'][1]
    T_limit_l, T_limit_r = motion['translate'][0], motion['translate'][1]
    
    

    for entry in meta:
        if entry['joint'] == 'hinge':  # Only consider hinge joints
            R_limit_l, R_limit_r = 0, entry['jointData']['limit']['b'] - entry['jointData']['limit']['a'] # entry['jointData']['limit']['b']
            arti_info = {
                'joint_id': entry['id'],
                'axis': {
                    'o': entry['jointData']['axis']['origin'],
                    'd': entry['jointData']['axis']['direction']
                },
                'rotate': {
                    'l': R_limit_l, # 'l': entry['jointData']['limit']['a'],  # start state from loaded articulation
                    'r': R_limit_r # entry['jointData']['limit']['b']   # end state from loaded articulation
                }
            }
            # arti_info = get_arti_info(entry, motions['motion']) 
        elif entry['joint'] == 'slider':
            T_limit_l, T_limit_r = entry['jointData']['limit']['a'] , entry['jointData']['limit']['b'] # entry['jointData']['limit']['b']
            # T_limit_l, T_limit_r = 0, abs(entry['jointData']['limit']['a'] - entry['jointData']['limit']['b']) # entry['jointData']['limit']['b']

            arti_info = {
                'joint_id': entry['id'],
                'axis': {
                    'o': entry['jointData']['axis']['origin'],
                    'd': entry['jointData']['axis']['direction']
                },
                'translate': {
                    'l': T_limit_l, # entry['jointData']['limit']['a'],  # start state from loaded articulation
                    'r': T_limit_r # entry['jointData']['limit']['b']   # end state from loaded articulation
                }
            }
        # else:
        #     raise NotImplemented(
        #         '{} joint is not implemented'.format(entry['joint']))

        articulations.append(arti_info)

    return articulations, meta  # return all articulations and the full metadata

def export_axis_mesh(arti, exp_dir):
    center = np.array(arti['axis']['o'], dtype=np.float32)
    k = np.array(arti['axis']['d'], dtype=np.float32)
    save_axis_mesh(k, center, os.path.join(exp_dir, 'axis_rotate.ply'))
    save_axis_mesh(-k, center, os.path.join(exp_dir, 'axis_rotate_oppo.ply'))

def generate_state(arti_info, meta, src_root, exp_dir, state):
    # joint_id = motions['joint_id']
    joint_id = arti_info['joint_id']
    motion_type = motions['motion']['type']
    
    ms = pymeshlab.MeshSet()
    ms_static = pymeshlab.MeshSet()
    ms_dynamic = pymeshlab.MeshSet()

    # 1. Load parts needs transformation to the mesh set
    for entry in meta:
        # add all moving parts into the meshset
        if entry['id'] == joint_id or entry['parent'] == joint_id:
            for mesh_fname in entry['visuals']:
                ms.load_new_mesh(os.path.join(src_root, mesh_fname))
                ms_dynamic.load_new_mesh(os.path.join(src_root, mesh_fname))


    # 2. Apply transformation
    if 'rotate' in arti_info:
        if state == 'start':
            degree = arti_info['rotate']['l']
        elif state == 'end':
            degree = arti_info['rotate']['r']
        elif state == 'canonical':
            degree = 0.5 * (arti_info['rotate']['r'] + arti_info['rotate']['l'])
        else:
            raise NotImplementedError
        # Filter: Transform: Rotate
        ms.compute_matrix_from_rotation(rotaxis='custom axis',
                                        rotcenter='custom point',
                                        angle=degree,
                                        customaxis=arti_info['axis']['d'],
                                        customcenter=arti_info['axis']['o'],
                                        snapflag=False,
                                        freeze=True,
                                        alllayers=True)
        ms_dynamic.compute_matrix_from_rotation(rotaxis='custom axis',
                                        rotcenter='custom point',
                                        angle=degree,
                                        customaxis=arti_info['axis']['d'],
                                        customcenter=arti_info['axis']['o'],
                                        snapflag=False,
                                        freeze=True,
                                        alllayers=True)
    elif 'translate' in arti_info:
        if state == 'start':
            dist = arti_info['translate']['l']
        elif state == 'end':
            dist = arti_info['translate']['r']
        elif state == 'canonical':
            dist = 0.5 * (arti_info['translate']['r'] + arti_info['translate']['l'])
        else:
            raise NotImplementedError

        # Filter: Transform: Translate, Center, set Origin
        ms.compute_matrix_from_translation_rotation_scale(
                                        translationx=arti_info['axis']['d'][0]*dist,
                                        translationy=arti_info['axis']['d'][1]*dist,
                                        translationz=arti_info['axis']['d'][2]*dist,
                                        alllayers=True)
        ms_dynamic.compute_matrix_from_translation_rotation_scale(
                                        translationx=arti_info['axis']['d'][0]*dist,
                                        translationy=arti_info['axis']['d'][1]*dist,
                                        translationz=arti_info['axis']['d'][2]*dist,
                                        alllayers=True)
    else:
        raise NotImplementedError

    # 3. load static parts to the mesh set
    for entry in meta:
        if entry['id'] != joint_id and entry['parent'] != joint_id:
            for mesh_fname in entry['visuals']:
                ms.load_new_mesh(os.path.join(src_root, mesh_fname))
                ms_static.load_new_mesh(os.path.join(src_root, mesh_fname))


    # 4. Merge Filter: Flatten Visible Layers
    ms, ms_static, ms_dynamic = merge_meshsets([ms, ms_static, ms_dynamic])


    # 5. Save original obj: y is up
    ms.save_current_mesh(os.path.join(exp_dir, f'{state}.obj'))

    # 6. Transform: Rotate, so that the object is at z-up frame
    mss = z_up_frame_meshsets([ms, ms_static, ms_dynamic])

    # 7. Save rotated meshes: z is up (align with the blender rendering)
    fnames = [
        os.path.join(exp_dir, f'{state}_rotate.ply'),
        os.path.join(exp_dir, f'{state}_static_rotate.ply'),
        os.path.join(exp_dir, f'{state}_dynamic_rotate.ply')
    ]
    save_meshsets_ply(mss, fnames)
    

def record_motion_json(motions, arti_info, dst_root):
    # coordinates changes from y-up to z-up
    axis_o = np.array(arti_info['axis']['o'])
    axis_d = np.array(arti_info['axis']['d'])
    R_coord = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    axis_o = np.matmul(R_coord, axis_o).tolist()
    axis_d = np.matmul(R_coord, axis_d).tolist()
    arti_info['axis']['o'] = axis_o
    arti_info['axis']['d'] = axis_d
    arti_info['type'] = motions['motion']['type']

    with open(os.path.join(dst_root, f'trans.json'), 'w') as f:
        conf = {
            'input': motions,
            'trans_info': arti_info
        }
        json.dump(conf, f)
        f.close()

    return arti_info

def generate_state_for_all_joints(articulations, meta, src_root, exp_dir, state):  # updated parameters
    ms = pymeshlab.MeshSet()
    ms_static = pymeshlab.MeshSet()
    ms_dynamic = pymeshlab.MeshSet()

    # 1. Load all parts into the mesh set
    # for entry in meta:
    #     for mesh_fname in entry['visuals']:
    #         ms.load_new_mesh(os.path.join(src_root, mesh_fname))
    #         if entry['joint'] == 'hinge' or entry['joint'] == 'slider':
    #             ms_dynamic.load_new_mesh(os.path.join(src_root, mesh_fname))
            # else:
            #     ms_static.load_new_mesh(os.path.join(src_root, mesh_fname))
            
    # for entry in meta:
    #     if entry['joint'] == 'hinge' or entry['joint'] == 'slider':
    #         for mesh_fname in entry['visuals']:
    #             ms.load_new_mesh(os.path.join(src_root, mesh_fname))
    #             ms_dynamic.load_new_mesh(os.path.join(src_root, mesh_fname))

    mesh_fnames_visited = set()
    # 2. Apply transformations for each articulation
    for arti_info in articulations:  # use the loaded articulations
        joint_id = arti_info['joint_id']

        
        # # 1. Load parts needs transformation to the mesh set
        for entry in meta:
            # add all moving parts into the meshset
            if (entry['id'] == joint_id or entry['parent'] == joint_id) and (entry['joint'] == 'hinge' or entry['joint'] == 'slider'):
                for mesh_fname in entry['visuals']:
                    if mesh_fname not in mesh_fnames_visited:
                        ms.load_new_mesh(os.path.join(src_root, mesh_fname))
                        ms_dynamic.load_new_mesh(os.path.join(src_root, mesh_fname))
                        mesh_fnames_visited.add(mesh_fname)
                    
                if 'rotate' in arti_info:
                    if state == 'start':
                        degree = arti_info['rotate']['l']
                    elif state == 'end':
                        degree = arti_info['rotate']['r']
                    elif state == 'canonical':
                        degree = 0.5 * (arti_info['rotate']['r'] + arti_info['rotate']['l'])
                    else:
                        raise NotImplementedError
                    # Filter: Transform: Rotate
                    ms.compute_matrix_from_rotation(rotaxis='custom axis',
                                                    rotcenter='custom point',
                                                    angle=degree,
                                                    customaxis=arti_info['axis']['d'],
                                                    customcenter=arti_info['axis']['o'],
                                                    snapflag=False,
                                                    freeze=True,
                                                    alllayers=True)
                    ms_dynamic.compute_matrix_from_rotation(rotaxis='custom axis',
                                                    rotcenter='custom point',
                                                    angle=degree,
                                                    customaxis=arti_info['axis']['d'],
                                                    customcenter=arti_info['axis']['o'],
                                                    snapflag=False,
                                                    freeze=True,
                                                    alllayers=True)
                    
                elif 'translate' in arti_info:
                    if state == 'start':
                        dist = 0 # arti_info['translate']['l']
                        # dist = motions['motion']['translate'][0]
                    elif state == 'end':
                        dist =  arti_info['translate']['r']
                        # dist = motions['motion']['translate'][1]
                    elif state == 'canonical':
                        dist = 0.5 * (arti_info['translate']['r'] + arti_info['translate']['l'])
                    else:
                        raise NotImplementedError

                    # Filter: Transform: Translate, Center, set Origin
                    ms.compute_matrix_from_translation_rotation_scale(
                                                    translationx=arti_info['axis']['d'][0]*dist,
                                                    translationy=arti_info['axis']['d'][1]*dist,
                                                    translationz=arti_info['axis']['d'][2]*dist,
                                                    alllayers=True)
                    ms_dynamic.compute_matrix_from_translation_rotation_scale(
                                                    translationx=arti_info['axis']['d'][0]*dist,
                                                    translationy=arti_info['axis']['d'][1]*dist,
                                                    translationz=arti_info['axis']['d'][2]*dist,
                                                    alllayers=True)
            
        # 3. load static parts to the mesh set
                # for entry in meta:
            elif entry['id'] != joint_id and entry['parent'] != joint_id:
                for mesh_fname in entry['visuals']:
                    if mesh_fname not in mesh_fnames_visited:
                        ms.load_new_mesh(os.path.join(src_root, mesh_fname))
                        ms_static.load_new_mesh(os.path.join(src_root, mesh_fname))
                        mesh_fnames_visited.add(mesh_fname)
                        
            
            
    # for entry in meta:
    #     if entry['joint'] != 'hinge' and entry['joint'] != 'slider':
    #         for mesh_fname in entry['visuals']:
    #             if mesh_fname not in mesh_fnames_visited:
    #                 ms.load_new_mesh(os.path.join(src_root, mesh_fname))
    #                 ms_static.load_new_mesh(os.path.join(src_root, mesh_fname))
        

    # 3. Merge Filter: Flatten Visible Layers
    ms, ms_static, ms_dynamic = merge_meshsets([ms, ms_static, ms_dynamic])

    # 4. Save original obj: y is up
    ms.save_current_mesh(os.path.join(exp_dir, f'{state}.obj'))

    # 5. Transform: Rotate, so that the object is at z-up frame
    mss = z_up_frame_meshsets([ms, ms_static, ms_dynamic])

    # 6. Save rotated meshes: z is up (align with the blender rendering)
    fnames = [
        os.path.join(exp_dir, f'{state}_rotate.ply'),
        os.path.join(exp_dir, f'{state}_static_rotate.ply'),
        os.path.join(exp_dir, f'{state}_dynamic_rotate.ply')
    ]
    save_meshsets_ply(mss, fnames)


def main(model_id, motions, src_root, dst_root):
    # states to be generated
    states = ['start', 'end']
    # create a json file with mesh info from urdf
    rewrite_json_from_urdf(src_root)

    # load articulations (y-up frame)
    # arti_info, meta = load_articulation(src_root, motions['joint_id'])
    articulations, meta = load_articulation(src_root)  # updated to retrieve all articulations

    
    # save meshes for each states
    for state in states:
        exp_dir = os.path.join(dst_root, state)
        os.makedirs(exp_dir, exist_ok=True)
        for arti_info in articulations:
            generate_state(arti_info, meta, src_root, exp_dir, state)
        # generate_state_for_all_joints(articulations, meta, src_root, exp_dir, state)  # updated call
        print(f'{state} done')

    # backup transformation json, convert articulation to z-up frame
    # arti = record_motion_json(motions, arti_info, dst_root)
    for arti_info in articulations:
        arti = record_motion_json(motions, arti_info, dst_root)

    # save mesh for motion axis
    export_axis_mesh(arti, dst_root)

if __name__ == '__main__':
    '''
    This script is to generate object mesh for each state.
    The articulation is referred to PartNet-Mobility <mobility_v2.json>
    '''
    
    # categories = ['Bottle', 'Dispenser', 'Kettle', 'Knife', 'Lamp', 'Lighter', 'Mouse', 'Pen', 'Pliers', 'Scissors', 'Toaster', 'USB']
    # data_path = '/media/qil/DATA/Carter_Articulated_Objects/paris-reconstruction/data/partnet-mobility/train'
    
    # # Traverse the directory structure
    # for category in os.listdir(data_path):
    #     category_path = os.path.join(data_path, category)
        
    #     if os.path.isdir(category_path):  # Check if it's a directory
    #         for model_id in os.listdir(category_path):
    #             model_id_path = os.path.join(category_path, model_id)
                
    #             if os.path.isdir(model_id_path):  # Check if it's a directory
    #                 # specify the export identifier (in this case, it's the same as model_id)
    #                 model_id_exp = model_id
    # specify the object category
    category = 'Knife'
    # specify the model id to be loaded
    model_id = '101217'     
    # data split
    split = 'train'
    # specify the export identifier
    model_id_exp = '101217'
    # specify the motion to generate new states
    motions = {
        'joint_id': 1, # joint id to be transformed (need to look up mobility_v2_self.json)
        'motion': {
            # type of motion expected: "rotate" or "translate"
            'type': 'rotate',   
            # range of the motion from start to end states
            'rotate': [0., 174.24.], 
            'translate': [0., 0.],
        },
    }

    # paths
    src_root = os.path.join(f'data/partnet-mobility/{split}/{category}', model_id)
    dst_root =  os.path.join(f'data/sapien_example/{category}', model_id_exp, 'textured_objs')

    main(model_id, motions, src_root, dst_root)


    
