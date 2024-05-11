import math
import os
import random

import torch
import numpy as np
import cv2
import pytorch_lightning as pl
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

from scene.cameras import Render_Camera
import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, rotmat2qvec
from utils.camera_utils import resize_mask_image, load_raw_depth
from utils.graphics_utils import getWorld2View2, focal2fov
from .random_camera_sampler import RandomCameraSampler
from scipy.spatial.transform import Rotation 

def myC2W_correct(R_c_to_w, T):
    """ returns camera to world matrix """
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R_c_to_w
    Rt[:3, 3] = R_c_to_w.matmul(-T)
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def find_tangent_vectors(normal):
    
    # Create a vector that is not parallel to the normal vector
    if normal[0] == 0 and normal[1] == 0:
        vec1 = np.cross(normal, np.array([1, 0, 0]))
    else:
        vec1 = np.cross(normal, np.array([0, 0, 1]))
    
    # Normalize the first tangent vector
    vec1 = vec1 / np.linalg.norm(vec1)
    
    # Use the cross product to find the second orthogonal vector in the tangent plane
    vec2 = np.cross(normal, vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    return vec1, vec2

def update_projection_matrix(viewpoint_cam, theta_x, theta_y, radius=10):
    """" description: 
    viewpoint.R: rotation matrix from camera to world coordinate system
    viewpoint.T: translation vector t=-R^T*t from world to camera (after rotation)
    viewpoint.camera_center: camera position in world coordinate system
    
    theta: rotation angle in radians
    radius: distance from the camera to the center of the rotation
    at: point in the world coordinate system that the camera is looking at (rotation point)
    
    Rotation:
    rotation performed relative to at point and around x-y arbitrary axis in the tangent plane at at's position
    
    USAGE: 
    M_ext, full_proj_transform, R_new, new_eye = update_projection_matrix(cam, theta_x=theta_x, theta_y=theta_y, radius=4)
    # 
    """
    # projection_matrix = viewpoint_cam.projection_matrix

    c2w = myC2W_correct(viewpoint_cam.R, viewpoint_cam.T)

    eye = c2w[:3, 3] # camera position in world coordinate system 

    # eye = c2w[:3, 3] - last row is the translation vector in world coordinate system after rotation t=-R^T*t (in code: R from camera to world)
    
    normal = c2w[:3, 2]/np.linalg.norm(c2w[:3, 2])
    
    at = eye + normal * radius
    
    rotation_vec = eye - at
    
    vec1, vec2 = find_tangent_vectors(normal)
    
    rotation_matrix = Rotation.from_rotvec(vec1*theta_x)
    rotation_matrix *= Rotation.from_rotvec(vec2*theta_y)
    rotated_vec = rotation_matrix.apply(rotation_vec)
    new_eye = at + rotated_vec
    R_new = rotation_matrix.as_matrix().T @ c2w[:3, :3] # NOTE: rotation_matrix.as_matrix() is in world2camera. R is in camera2world
    R_new = rotation_matrix.as_matrix() @ c2w[:3, :3]
    
    T = -R_new.T @ new_eye
    world_view_transform = torch.tensor(getWorld2View2(R_new, T, translate=viewpoint_cam.trans.numpy(), scale=viewpoint_cam.scale)).transpose(0, 1).cuda()
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0)).squeeze(0)

    # convert to torch with original dtype and device 
    new_eye = torch.tensor(new_eye, dtype=viewpoint_cam.camera_center.dtype, device=viewpoint_cam.camera_center.device)
    return world_view_transform, full_proj_transform, R_new, new_eye, T

def getNerfppNorm(cam_centers):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate.astype(np.float32), "radius": float(radius)}


@dataclass
class LooDataModuleConfig:
    batch_size: int = 1
    data_dir: str = ''
    eval_camera_distance: float = 6.
    resolution: int = 1
    prompt: str = ''
    sparse_num: int = 0
    bg_white: bool = False
    length: int = 1500
    around_gt_steps: int = 750
    refresh_interval: int = 100
    refresh_size: int = 20


@register("loo-dataset")
class LooDataset(Dataset):
    def __init__(self, cfg: LooDataModuleConfig, split: str = 'train', sparse_num: int = 0):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.data_dir = self.cfg.data_dir # data/helmet2
        self.resolution = self.cfg.resolution
        self.sparse_num = sparse_num
        self.length = self.cfg.length # 4000
        self.around_gt_steps = self.cfg.around_gt_steps # 2800
        self.refresh_interval = self.cfg.refresh_interval # 200
        self.refresh_size = self.cfg.refresh_size # 8
        self.gt_cameras = []
        
        self.sparse_ids = []
        if self.sparse_num != 0:
            if self.split == 'train':
                with open(os.path.join(self.data_dir, f'sparse_{self.sparse_num}.txt')) as f:
                    self.sparse_ids = sorted([int(id) for id in f.readlines()])
            else:
                with open(os.path.join(self.data_dir, f'sparse_test.txt')) as f:
                    self.sparse_ids = sorted([int(id) for id in f.readlines()])

        cameras_extrinsic_file = os.path.join(self.data_dir, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(self.data_dir, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file, selected_ids=self.sparse_ids)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        cam_extrinsics_unsorted = list(cam_extrinsics.values())
        cam_extrinsics = sorted(cam_extrinsics_unsorted.copy(), key = lambda x : x.name)

        

        images_folder=os.path.join(self.data_dir, 'images')
        if self.resolution in [1, 2, 4, 8]:
            tmp_images_folder = images_folder + f'_{str(self.resolution)}' if self.resolution != 1 else images_folder
            if not os.path.exists(tmp_images_folder):
                threestudio.warn(f"The {tmp_images_folder} is not found, use original resolution images")
            else:
                threestudio.info(f"Using resized images in {tmp_images_folder}...")
                images_folder = tmp_images_folder
        else:
            threestudio.info("use original resolution images")
        masks_folder = os.path.join(self.data_dir, 'masks')

        self.Rs, self.Ts, self.heights, self.widths, self.fovxs, self.fovys, self.images, self.masks, self.depths = [], [], [], [], [], [], [], [], []
        cam_c = []
        for extr in cam_extrinsics:
            img_file_num = int(
                "".join([s for s in os.path.basename(extr.name) if s.isdigit()])
                )
            if img_file_num in self.sparse_ids:
                intr = cam_intrinsics[extr.camera_id]
                height = intr.height
                width = intr.width

                R = np.transpose(qvec2rotmat(extr.qvec))
                T = np.array(extr.tvec)

                if intr.model=="SIMPLE_PINHOLE":
                    focal_length_x = intr.params[0]
                    FovY = focal2fov(focal_length_x, height)
                    FovX = focal2fov(focal_length_x, width)
                elif intr.model=="PINHOLE":
                    focal_length_x = intr.params[0]
                    focal_length_y = intr.params[1]
                    FovY = focal2fov(focal_length_y, height)
                    FovX = focal2fov(focal_length_x, width)
                else:
                    assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

                cam_c.append(np.linalg.inv(getWorld2View2(R, T))[:3, 3:4])
                self.Rs.append(R)
                self.Ts.append(T)
                self.fovxs.append(FovX)
                self.fovys.append(FovY)

                image_path = os.path.join(images_folder, os.path.basename(extr.name))
                image_name = os.path.basename(image_path).split(".")[0]
                image = Image.open(image_path)

                mask_path = os.path.join(masks_folder, image_name + '.png')
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                mask = mask.astype(np.float32) / 255.0
                resized_mask = resize_mask_image(mask, image.size)
                loaded_mask = torch.from_numpy(resized_mask).unsqueeze(0)

                depth_path = os.path.join(os.path.dirname(images_folder), "zoe_depth", os.path.basename(
                    image_path).replace(os.path.splitext(os.path.basename(image_path))[-1], '.png'))
                depth = load_raw_depth(depth_path)
                resized_depth = cv2.resize(depth, image.size, interpolation=cv2.INTER_NEAREST)
                loaded_depth = torch.from_numpy(resized_depth).unsqueeze(0)
                loaded_depth[loaded_mask <= 0.5] = 0.

                # mask image
                image = (torch.from_numpy(np.array(image))/255.).permute(2, 0, 1) # C, H, W
                image[(loaded_mask <= 0.5).expand_as(image)] = 1.0 if self.cfg.bg_white else 0.0

                self.images.append(image)
                self.masks.append(loaded_mask.squeeze())
                self.depths.append(loaded_depth.squeeze())

                self.heights.append(image.shape[-2])
                self.widths.append(image.shape[-1])
                self.gt_cameras.append(Render_Camera(torch.from_numpy(R), torch.from_numpy(T), FovX, FovY, image, loaded_mask, loaded_depth, white_background = True))

        all_Rs = []
        all_Ts = []
        cam_c = []
        for extr in cam_extrinsics:
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

            cam_c.append(np.linalg.inv(getWorld2View2(R, T))[:3, 3:4])
            all_Rs.append(R)
            all_Ts.append(T)

        self.cameras_extent = getNerfppNorm(cam_c)
        self.camera_sampler = RandomCameraSampler(self.Rs, self.Ts, all_Rs, all_Ts)

        self.cnt = 0
        self.random_poses = []

    def refresh_random_poses(self):
        self.random_poses = []
        self.our_random_poses = []
        cnt = 0

        dis_from_gt = 0.8
        threestudio.info(f'refresh random poses with dis_drom_gt={dis_from_gt} at step {self.cnt}')
        self.random_poses = []
        while len(self.random_poses) < self.refresh_size: # refresh_size=8
            samples = self.camera_sampler.sample_away_from_gt(dis_from_gt) # list of len gt, each element is a tuple of (R,T), with (3,3) and (3,)
            self.random_poses.extend(samples)
            # our update
            for i in range(len(self.gt_cameras)):
                gt_camera = self.gt_cameras[i]
                theta_x = np.random.uniform(-np.pi/10, np.pi/10)
                theta_y = np.random.uniform(-np.pi/10, np.pi/10)
                world_view_transform, full_proj_transform, R_new, new_eye, T_new = update_projection_matrix(gt_camera, theta_x, theta_y, radius=4)
                self.our_random_poses.append((R_new, T_new))
        self.random_poses = self.random_poses[:self.refresh_size] # list of len self.refresh_size, each element is a tuple of (R,T), with (3,3) and (3,)
        self.our_random_poses = self.our_random_poses[:self.refresh_size]

    def __len__(self):
        if self.split == 'train':
            return self.cfg.length
        elif len(self.sparse_ids):
            return len(self.sparse_ids)
        else:
            return len(self.Rs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == 'train':
            idx = random.randint(0, len(self.sparse_ids) - 1)
            random_index = random.randint(0, self.refresh_size - 1)
            if self.cnt < self.around_gt_steps: # 2800
                if self.cnt % self.cfg.refresh_interval == 0:
                    self.refresh_random_poses() # self.random_poses will contain len refresh_size random poses - each pose a tuple (R,T), where R(3,3), T (3,)
                random_R, random_T = self.random_poses[random_index]
            else:
                random_R, random_T = self.camera_sampler.sample(None)
            self.cnt += 1
        else:
            theta = 2 * math.pi * idx / len(self)
            random_index = idx
            random_R, random_T = self.camera_sampler.sample(theta)
        ret = {
            "index": idx,
            "R": self.Rs[idx],
            "T": self.Ts[idx],
            "height": self.heights[idx],
            "width": self.widths[idx],
            "fovx": self.fovxs[idx],
            "fovy": self.fovys[idx],
            "image": self.images[idx],
            "mask": self.masks[idx],
            "depth": self.depths[idx],
            "txt": self.cfg.prompt,
            "random_index": random_index,
            "random_R": random_R,
            "random_T": random_T,
            "random_poses": self.random_poses,
            "gt_images": self.images,
            "gt_Ts": self.Ts,
            "our_random_poses": self.our_random_poses
        }
        return ret

    def get_scene_extent(self):
        return self.cameras_extent

    def norm_to_pc(self, center):
        self.Ts = [(T - center) for T in self.Ts]


@register("loo-datamodule")
class LooDataModuleFromConfig(pl.LightningDataModule):
    cfg: LooDataModuleConfig
    train_dataset: Optional[LooDataset] = None
    val_dataset: Optional[LooDataset] = None
    test_dataset: Optional[LooDataset] = None

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(LooDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = LooDataset(self.cfg, "train", sparse_num=self.cfg.sparse_num)
        # if stage in [None, "fit", "validate"]:
        #     self.val_dataset = LooDataset(self.cfg, "val", sparse_num=self.cfg.sparse_num)
        if stage in [None, "test", "predict"]:
            self.test_dataset = LooDataset(self.cfg, "test", sparse_num=self.cfg.sparse_num)

    def norm_to_pc(self, center):
        if self.train_dataset is not None:
            self.train_dataset.norm_to_pc(center)
        if self.val_dataset is not None:
            self.val_dataset.norm_to_pc(center)
        if self.test_dataset is not None:
            self.test_dataset.norm_to_pc(center)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )
