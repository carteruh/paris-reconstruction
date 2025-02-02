import torch
import torch.nn as nn
from math import sin, cos
import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching
from nerfacc.vol_rendering import render_transmittance_from_alpha, rendering
from utils.rotation import quaternion_to_axis_angle, R_from_axis_angle, R_from_quaternions

@models.register('revolute')
class RevoluteModel(BaseModel):
    def setup(self):
        self.static_geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.static_texture = models.make(self.config.texture.name, self.config.texture)
        self.dynamic_geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.dynamic_texture = models.make(self.config.texture.name, self.config.texture)

        # initialize the motion parameters
        init_angle = self.config.get('init_angle', 0.1)
        init_dir = self.config.get('init_dir', [1., 1., 1.])
        self.axis_o = nn.Parameter(torch.tensor([0., 0., 0.], dtype=torch.float32), requires_grad=True)
        self.quaternions = nn.Parameter(self.init_quaternions(half_angle=init_angle, init_dir=init_dir), requires_grad=True) # real part first
 
        self.canonical = 0.5
        self.use_part_mask = self.config.get('use_part_mask', False)

        # configurations for ray marching speed up
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.grid_warmup = self.config['grid_warmup']
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128, 
                contraction_type=ContractionType.AABB
            )
        self.randomized = self.config.randomized
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        if self.config.white_bkgd:
            self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
            self.background_color.to(self.rank)
    
    def forward_(self, rays, scene_state):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def sigma_fn_composite(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            sigma_s, _ = self.static_geometry(positions)
            positions = self.rigid_transform(positions, scene_state)
            sigma_d, _ = self.dynamic_geometry(positions)
            sigma = sigma_s + sigma_d
            return sigma[...,None]
    
        def rgb_sigma_fn_static(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, feature = self.static_geometry(positions) 
            rgb = self.static_texture(feature, t_dirs)
            return rgb, density[...,None]
        
        def rgb_sigma_fn_dynamic(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            positions = self.rigid_transform(positions, scene_state)
            density, feature = self.dynamic_geometry(positions) 
            dirs_d = self.rigid_transform(t_dirs, scene_state)
            rgb = self.dynamic_texture(feature, dirs_d)
            return rgb, density[...,None]

        def composite_rendering(ray_indices, t_starts, t_ends):
            n_rays = rays_o.shape[0]
            rgb_s, sigma_s = rgb_sigma_fn_static(t_starts, t_ends, ray_indices)
            rgb_d, sigma_d = rgb_sigma_fn_dynamic(t_starts, t_ends, ray_indices)

            dists = t_ends - t_starts
            alpha_s = 1. - torch.exp(-sigma_s * dists)
            alpha_d = 1. - torch.exp(-sigma_d * dists)

            alpha_add = 1. - (1. - alpha_s) * (1. - alpha_d)
            Ts = render_transmittance_from_alpha(alpha_add, ray_indices=ray_indices)

            weights_s = alpha_s * Ts
            weights_d = alpha_d * Ts
            
            weights = weights_s + weights_d
            # opacity
            opacity = self.acc_along_rays(weights, ray_indices, n_rays)
            opacity = opacity.squeeze(-1)
            # acc color
            rgb = weights_s * rgb_s + weights_d * rgb_d
            rgb = self.acc_along_rays(rgb, ray_indices, n_rays)
            # Background composition.
            if self.config.white_bkgd:
                rgb = rgb + self.background_color * (1. - opacity[..., None])

            # regularization
            ratio = (sigma_s > 1.).detach() * sigma_d.detach() / torch.clamp_min(sigma_s + sigma_d.detach(), 1e-10) # only reg static      

            # validation and testing
            if not self.training:
                # depth
                depth = weights * ((t_starts + t_ends) * 0.5)
                depth = self.acc_along_rays(depth, ray_indices, n_rays)
                depth = depth.squeeze(-1)

                rgb_s_only, opacity_s, depth_s_only = rendering(t_starts, t_ends, ray_indices, n_rays,
                                                                rgb_sigma_fn=rgb_sigma_fn_static, 
                                                                render_bkgd=self.background_color)
                rgb_d_only, opacity_d, depth_d_only = rendering(t_starts, t_ends, ray_indices, n_rays,
                                                                rgb_sigma_fn=rgb_sigma_fn_dynamic, 
                                                                render_bkgd=self.background_color)
                return {
                    'rgb': rgb, 
                    'opacity': opacity, 
                    'depth': depth,  
                    'rgb_s': rgb_s_only, 
                    'rgb_d': rgb_d_only, 
                    'depth_s': depth_s_only, 
                    'depth_d': depth_d_only, 
                    'opacity_s': opacity_s, 
                    'opacity_d': opacity_d, 
                }
            
            # regularization on part mask
            part_mask_ratio = None
            if self.use_part_mask:
                opacity_s = self.acc_along_rays(weights_s, ray_indices, n_rays)
                opacity_d = self.acc_along_rays(weights_d, ray_indices, n_rays)
                part_mask_ratio = opacity_s / torch.clamp_min(opacity_s + opacity_d, 1e-10)
            
            return {
                'rgb': rgb, 
                'ratio': ratio.squeeze(-1), 
                'rays_valid': opacity > 0, 
                'opacity': opacity,
                'part_mask_ratio': part_mask_ratio,
            }

        with torch.no_grad():
            # nerfacc 0.3.2
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                sigma_fn=sigma_fn_composite,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
            )
        render_out = composite_rendering(ray_indices, t_starts, t_ends)

        if self.training:
            return {
                'comp_rgb': render_out['rgb'],
                'ratio': render_out['ratio'],
                'opacity': render_out['opacity'],
                'rays_valid': render_out['rays_valid'],
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
                'part_mask_ratio': render_out['part_mask_ratio'],
            }

        return {
            'comp_rgb': render_out['rgb'],
            'opacity': render_out['opacity'],
            'depth': render_out['depth'],
            'rgb_s': render_out['rgb_s'],
            'rgb_d': render_out['rgb_d'],
            'depth_s': render_out['depth_s'],
            'depth_d': render_out['depth_d'],
            'opacity_s': render_out['opacity_s'],
            'opacity_d': render_out['opacity_d'],
        }


    def forward(self, rays_0, rays_1):
        if self.training:
            out_0 = self.forward_(rays_0, scene_state=0.)
            out_1 = self.forward_(rays_1, scene_state=1.)
        else:
            out_0 = chunk_batch(self.forward_, self.config.ray_chunk, rays_0, scene_state=0.)
            out_1 = chunk_batch(self.forward_, self.config.ray_chunk, rays_1, scene_state=1.)
            del rays_0, rays_1
        return [{**out_0}, {**out_1}]

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def isosurface(self):
        mesh_s = self.static_geometry.isosurface()
        mesh_d = self.dynamic_geometry.isosurface()
        return {'static': mesh_s, 'dynamic': mesh_d}

    
    def init_quaternions(self, half_angle, init_dir):
        a = torch.tensor([init_dir[0], init_dir[1], init_dir[2]], dtype=torch.float32)
        a = torch.nn.functional.normalize(a, p=2., dim=0)
        sin_ = sin(half_angle)
        cos_ = cos(half_angle)
        r = cos_
        i = a[0] * sin_
        j = a[1] * sin_
        k = a[2] * sin_
        q = torch.tensor([r, i, j, k], dtype=torch.float32)
        return q
    
   
    def rigid_transform(self, positions, state=0.):
        '''
        Perform the rigid transformation: R @ (x - c) + c

        Transform the positions from canonical state=0.5 to state=0 or state=1
        '''
        
        scaling = (self.canonical - state) / self.canonical

        positions = positions - self.axis_o
        if scaling == 1.:
            R = R_from_quaternions(self.quaternions)
            positions = torch.matmul(R, positions.T).T
        elif scaling == -1.:
            inv_sc = torch.tensor([1., -1., -1., -1]).to(self.quaternions)
            inv_q = inv_sc * self.quaternions
            R = R_from_quaternions(inv_q)
            positions = torch.matmul(R, positions.T).T
        else:
            axis, angle = quaternion_to_axis_angle(self.quaternions) # the angle means from t=0 to t=0.5
            tgt_angle = scaling * angle
            R = R_from_axis_angle(axis, tgt_angle)
            positions = torch.matmul(R, positions.T).T
        positions = positions + self.axis_o

        return positions
    
    
    def regularizations(self, outs):
        losses = {}
        return losses
    
    def update_step(self, epoch, global_step):
        ''' This function is not really used'''
        # progressive viewdir PE frequencies
        update_module_step(self.static_texture, epoch, global_step)
        update_module_step(self.dynamic_texture, epoch, global_step)
        
        def occ_eval_fn(x):
            density_s, _ = self.static_geometry(x)
            x_d = self.rigid_transform(x)
            density_d, _ = self.dynamic_geometry(x_d)
            density = density_s + density_d
            return density[...,None] * self.render_step_size
        
        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=1e-4, warmup_steps=self.grid_warmup)

    
   