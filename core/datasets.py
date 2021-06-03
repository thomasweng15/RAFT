# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

import cv2
from flow import GTFlow, remove_dups
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image

class TowelTest(data.Dataset):
    def __init__(self, config, camera_params, foldtype='unfold', datatype='sim'):
        self.camera_params = camera_params
        self.datatype = datatype
        self.flw = GTFlow()
        # self.plot = plot

        self.eval_combos = [['open_2side_high', f'towel_train_{i}_high'] for i in range(32)]

        # if foldtype == 'all':
        #     self.eval_combos = [
        #         ['open_2side', 'one_corner_in_2side'],
        #         ['open_2side', 'opp_corners_in_2side'],
        #         ['open_2side', 'all_corners_in_2side'],
        #         ['open_2side', 'triangle_2side'],
        #         ['one_corner_in_2side', 'open_2side'],
        #         ['opp_corners_in_2side', 'open_2side'],
        #         ['all_corners_in_2side', 'open_2side'],
        #         ['triangle_2side', 'open_2side']]
        # elif foldtype == 'fold':
        #     self.eval_combos = [
        #         ['open_2side', 'one_corner_in_2side'],
        #         ['open_2side', 'opp_corners_in_2side'],
        #         ['open_2side', 'all_corners_in_2side'],
        #         ['open_2side', 'triangle_2side']]
        # elif foldtype == 'unfold':
        #     self.eval_combos = [
        #         ['one_corner_in_2side', 'open_2side'],
        #         ['opp_corners_in_2side', 'open_2side'],
        #         ['all_corners_in_2side', 'open_2side'],
        #         ['triangle_2side', 'open_2side']]
    
    def __len__(self):
        return len(self.eval_combos)

    # def load_depth(self, name):
    #     if self.datatype == 'sim':
    #         depth = np.load(f'/data/fabric_data/sim2real/real_gray/{name}_gray.npy')[0] / 1000
    #         depth = depth[90:-80, 150:-180].astype(np.float32) # 310 x 310
    #         mask = (depth == 0).astype(np.uint8)
    #         depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
    #         depth = cv2.resize(depth, (200, 200))
    #     elif self.datatype == 'real':
    #         depth = np.load(f'/data/fabric_data/sim2real/real_gray/{name}_gray.npy')[0] / 1000
    #         depth = depth[90:-80, 150:-180].astype(np.float32) # 310 x 310
    #         mask = (depth == 0).astype(np.uint8)
    #         depth = cv2.inpaint(depth, mask, 3, cv2.INPAINT_NS)
    #         depth = cv2.resize(depth, (200, 200))
    #     elif self.datatype == 'gan':
    #         depth = cv2.imread(f'/data/fabric_data/cyclegan/testB_output/clothcyclegan_maskdrpng/test_latest/images/{name}_gray_fake.png')[:, :, 0] # 256 x 256
    #         depth = depth / 255.0
    #         depth = cv2.resize(depth, (200, 200))
    #         # import IPython; IPython.embed()
    #     return depth

    # def __getitem__(self, index):
    #     start_fn, goal_fn = self.eval_combos[index]
    #     depth_o = self.load_depth(start_fn)
    #     depth_n = self.load_depth(goal_fn)

    #     # Load mask

    #     return depth_o, depth_n

    def __getitem__(self, index):
        start_fn, goal_fn = self.eval_combos[index]
        path = '/home/exx/projects/softagent/descriptors_softgym_baseline'

        depth_o = cv2.imread(f'{path}/goals/{start_fn}_depth.png')[:, :, 0] / 255 # 200 x 200
        cloth_mask = (depth_o != 0).astype(float) # 200 x 200

        if not os.path.exists(f'{path}/goals/particles/{start_fn}_uvnodups.npy'):
            coords_o = np.load(f'{path}/goals/particles/{start_fn}.npy')
            uv_o_f = np.load(f'{path}/goals/particles/{start_fn}_uv.npy')
            # if self.cfg['dataname'] == 'sg_towel_actcorlbaseline_n2200_edgethresh5_actmask0.9_ceratio0.5_ontable0':
            # uv_o_f[:,[1,0]] = uv_o_f[:,[0,1]] # knots axes are flipped in collect_data
            
            # Remove occlusions
            depth_o_rs = cv2.resize(depth_o, (720, 720))
            uv_o = remove_dups(self.camera_params, uv_o_f, coords_o, depth_o_rs, zthresh=0.005)
            np.save(f'{path}/goals/particles/{start_fn}_uvnodups.npy', uv_o)
        else:
            uv_o = np.load(f'{path}/goals/particles/{start_fn}_uvnodups.npy')
        
        # Load nobs and knots
        # With probablity p, sample image pair as obs and nobs, otherwise choose random nobs
        depth_n = cv2.imread(f'{path}/goals/{goal_fn}_depth.png')[:, :, 0] / 255
        uv_n_f = np.load(f'{path}/goals/particles/{goal_fn}_uv.npy')
        # if self.cfg['dataname'] == 'sg_towel_actcorlbaseline_n2200_edgethresh5_actmask0.9_ceratio0.5_ontable0':
        # uv_n_f[:,[1,0]] = uv_n_f[:,[0,1]] # knots axes are flipped in collect_data

        # Remove out of bounds
        uv_o[uv_o < 0] = float('NaN')
        uv_o[uv_o >= 720] = float('NaN')

         # Get flow image
        flow_lbl = self.flw.get_image(uv_o, uv_n_f, mask=cloth_mask, depth_o=depth_o, depth_n=depth_n)

        # Get loss mask
        valid = np.zeros((flow_lbl.shape[0], flow_lbl.shape[1]), dtype=np.float32)
        non_nan_idxs = np.rint(uv_o[~np.isnan(uv_o).any(axis=1)]/719*199).astype(int)
        valid[non_nan_idxs[:, 0], non_nan_idxs[:, 1]] = 1

        if False:
            im1 = depth_o
            im2 = depth_n
            # flow_im = flow_pr.squeeze().permute(1, 2, 0).cpu().numpy()
            # mask = im1 == 0
            # flow_im[mask, :] = 0
            fig, ax = plt.subplots(1, 4, figsize=(16, 8))
            ax[0].imshow(im1)
            ax[1].imshow(im2)
            
            skip = 1
            h, w, _ = flow_lbl.shape
            ax[2].imshow(np.zeros((h, w)), alpha=0.5)
            ys, xs, _ = np.where(flow_lbl != 0)
            ax[2].quiver(xs[::skip], ys[::skip],
                        flow_lbl[ys[::skip], xs[::skip], 1], flow_lbl[ys[::skip], xs[::skip], 0], 
                        alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

            ax[3].imshow(valid)

            # skip = 1
            # flow_gt = flow_lbl
            # # flow_gt = flow_gt.permute(1, 2, 0).numpy()
            # ax[3].imshow(np.zeros((h, w)), alpha=0.5)
            # ys, xs, _ = np.where(flow_gt != 0)
            # ax[3].quiver(xs[::skip], ys[::skip],
            #             flow_gt[ys[::skip], xs[::skip], 1], flow_gt[ys[::skip], xs[::skip], 0], 
            #             alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

            plt.tight_layout()
            plt.show()

        depth1 = torch.from_numpy(depth_o).unsqueeze(2).permute(2, 0, 1).float()
        depth2 = torch.from_numpy(depth_n).unsqueeze(2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow_lbl).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()
        return depth1, depth2, flow, valid
        

class Towel(data.Dataset):
    def __init__(self, cfg, ids, camera_params, aug_params=None, sparse=True, spatialaug=False, switchobs=False, stage='train'):
        self.augmentor = None
        self.sparse = sparse
        self.spatialaug = spatialaug
        self.switchobs = switchobs
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []

        self.camera_params = camera_params
        self.cfg = cfg
        dataname = 'dataname' if stage=='train' else 'valname'
        self.data_path = f"{self.cfg['basepath']}/{self.cfg[dataname]}"
        self.transform = T.Compose([T.ToTensor()])
        self.flw = GTFlow()

        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = self.ids[index]

        switch = self.switchobs and torch.rand(1) < 0.5
        obs_suffix = 'after' if switch else 'before'
        nobs_suffix = 'before' if switch else 'after'

        # Load obs and knots
        depth_o = np.load(f'{self.data_path}/rendered_images/{str(index).zfill(6)}_depth_{obs_suffix}.npy')
        cloth_mask = (depth_o != 0).astype(float) # 200 x 200

        if not os.path.exists(f'{self.data_path}/knots/{str(index).zfill(6)}_knotsnodups_{obs_suffix}.npy'):
            coords_o = np.load(f'{self.data_path}/coords/{str(index).zfill(6)}_coords_{obs_suffix}.npy')
            uv_o_f = np.load(f'{self.data_path}/knots/{str(index).zfill(6)}_knots_{obs_suffix}.npy')
            # if self.cfg['dataname'] == 'sg_towel_actcorlbaseline_n2200_edgethresh5_actmask0.9_ceratio0.5_ontable0':
            uv_o_f[:,[1,0]] = uv_o_f[:,[0,1]] # knots axes are flipped in collect_data
            
            # Remove occlusions
            depth_o_rs = cv2.resize(depth_o, (720, 720))
            uv_o = remove_dups(self.camera_params, uv_o_f, coords_o, depth_o_rs, zthresh=0.001)
            np.save(f'{self.data_path}/knots/{str(index).zfill(6)}_knotsnodups_{obs_suffix}.npy', uv_o)
        else:
            uv_o = np.load(f'{self.data_path}/knots/{str(index).zfill(6)}_knotsnodups_{obs_suffix}.npy')

        # Load nobs and knots
        # With probablity p, sample image pair as obs and nobs, otherwise choose random nobs
        depth_n = np.load(f'{self.data_path}/rendered_images/{str(index).zfill(6)}_depth_{nobs_suffix}.npy')
        uv_n_f = np.load(f'{self.data_path}/knots/{str(index).zfill(6)}_knots_{nobs_suffix}.npy')
        # if self.cfg['dataname'] == 'sg_towel_actcorlbaseline_n2200_edgethresh5_actmask0.9_ceratio0.5_ontable0':
        uv_n_f[:,[1,0]] = uv_n_f[:,[0,1]] # knots axes are flipped in collect_data

        # Spatial aug
        if self.spatialaug and torch.rand(1) < 0.9:
            depth_o = Image.fromarray(depth_o)
            depth_n = Image.fromarray(depth_n)
            cloth_mask = Image.fromarray(cloth_mask)
            depth_o, depth_n, cloth_mask, uv_o, uv_n_f = self.spatial_aug(depth_o, depth_n, cloth_mask, uv_o, uv_n_f)
            depth_o = np.array(depth_o)
            depth_n = np.array(depth_n)
        cloth_mask = np.array(cloth_mask, dtype=bool)

        # Remove out of bounds
        uv_o[uv_o < 0] = float('NaN')
        uv_o[uv_o >= 720] = float('NaN')

        # Get flow image
        flow_lbl = self.flw.get_image(uv_o, uv_n_f, mask=cloth_mask, depth_o=depth_o, depth_n=depth_n)

        # Get loss mask
        valid = np.zeros((flow_lbl.shape[0], flow_lbl.shape[1]), dtype=np.float32)
        non_nan_idxs = np.rint(uv_o[~np.isnan(uv_o).any(axis=1)]/719*199).astype(int)
        valid[non_nan_idxs[:, 0], non_nan_idxs[:, 1]] = 1

        if False:
            im1 = depth_o
            im2 = depth_n
            # flow_im = flow_pr.squeeze().permute(1, 2, 0).cpu().numpy()
            # mask = im1 == 0
            # flow_im[mask, :] = 0
            fig, ax = plt.subplots(1, 4, figsize=(16, 8))
            ax[0].imshow(im1)
            ax[1].imshow(im2)
            
            skip = 1
            h, w, _ = flow_lbl.shape
            ax[2].imshow(np.zeros((h, w)), alpha=0.5)
            ys, xs, _ = np.where(flow_lbl != 0)
            ax[2].quiver(xs[::skip], ys[::skip],
                        flow_lbl[ys[::skip], xs[::skip], 1], flow_lbl[ys[::skip], xs[::skip], 0], 
                        alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

            ax[3].imshow(valid)

            plt.tight_layout()
            plt.show()

        depth1 = torch.from_numpy(depth_o).unsqueeze(2).permute(2, 0, 1).float()
        depth2 = torch.from_numpy(depth_n).unsqueeze(2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow_lbl).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()
        return depth1, depth2, flow, valid

    def aug_uv(self, uv, angle, dx, dy):
        uvt = deepcopy(uv)
        rad = np.deg2rad(angle)
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]])
        uvt -= 719 / 2
        uvt = np.dot(R, uvt.T).T
        uvt += 719 / 2
        uvt[:, 1] += dx
        uvt[:, 0] += dy
        return uvt

    def spatial_aug(self, depth_o, depth_n, cloth_mask, uv_o, uv_n_f):
        angle = int(torch.randint(low=-5, high=6, size=(1,)).numpy()[0])
        dx = int(torch.randint(low=-5, high=6, size=(1,)).numpy()[0])
        dy = int(torch.randint(low=-5, high=6, size=(1,)).numpy()[0])
        depth_o = TF.affine(depth_o, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        depth_n = TF.affine(depth_n, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        cloth_mask = TF.affine(cloth_mask, angle=angle, translate=(dx, dy), scale=1.0, shear=0)
        uv_o = self.aug_uv(uv_o, -angle, dx/199*719, dy/199*719)
        uv_n_f = self.aug_uv(uv_n_f, -angle, dx/199*719, dy/199*719)
        return depth_o, depth_n, cloth_mask, uv_o, uv_n_f

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


def fetch_dataloader(args, cfg, ids, camera_params, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'towel':
        # aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        aug_params = None
        train_dataset = Towel(cfg, ids, camera_params, aug_params, spatialaug=args.spatial_aug, switchobs=args.switchobs, stage='train')

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=args.n_workers, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

