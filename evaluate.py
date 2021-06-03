import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

import yaml
import random
from flow import FlowNetSmall
import time

@torch.no_grad()
def create_sintel_submission(model, iters=32, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)
        
        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            flow_low, flow_pr = model(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()
            
            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(model, iters=24, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs': epe}

@torch.no_grad()
def validate_sintel(model, iters=32):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}

@torch.no_grad()
def validate_towel(model, cfg, ids, camera_params, iters=24, plot=False, flownet_model='', dataset='towel'):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    if flownet_model != '':
        flownet = FlowNetSmall(input_channels=2).cuda()
        flownet.load_state_dict(torch.load(flownet_model))
        flownet.eval()
        epe1_list = []

    if dataset == 'towel':
        val_dataset = datasets.Towel(cfg, ids, camera_params, spatialaug=False, switchobs=False, stage='val')
    elif dataset == 'toweltest':
        val_dataset = datasets.TowelTest(cfg, camera_params)

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        val = valid_gt.view(-1) >= 0.5
        if flownet_model:
            flow_inp = torch.tensor(np.stack([image1, image2], axis=1), dtype=torch.float32, device='cuda')
            flow_pr1 = flownet(flow_inp)
            epe1 = torch.sum((flow_pr1[0].cpu() - flow_gt)**2, dim=0).sqrt()
            epe1 = epe1.view(-1)
            epe1_list.append(epe1[val].mean().numpy())

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe = epe.view(-1)
        epe_list.append(epe[val].mean().numpy())

        if plot:
            im1 = image1.squeeze().cpu().numpy()
            im2 = image2.squeeze().cpu().numpy()
            flow_im = flow_pr.squeeze().permute(1, 2, 0).cpu().numpy()
            mask = im1 == 0
            flow_im[mask, :] = 0
            
            n = 5 if flownet_model else 4
            fig, ax = plt.subplots(1, n, figsize=(16, 8))
            ax[0].imshow(im1)
            ax[1].imshow(im2)
            
            skip = 15
            h, w, _ = flow_im.shape
            ax[2].set_title(f"RAFT EPE: {epe[val].mean().numpy():0.3f}")
            ax[2].imshow(np.zeros((h, w)), alpha=0.5)
            ys, xs, _ = np.where(flow_im != 0)
            ax[2].quiver(xs[::skip], ys[::skip],
                        flow_im[ys[::skip], xs[::skip], 1], flow_im[ys[::skip], xs[::skip], 0], 
                        alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

            skip = 1
            ax[3].set_title(f"Ground truth")
            flow_gt = flow_gt.permute(1, 2, 0).numpy()
            ax[3].imshow(np.zeros((h, w)), alpha=0.5)
            ys, xs, _ = np.where(flow_gt != 0)
            ax[3].quiver(xs[::skip], ys[::skip],
                        flow_gt[ys[::skip], xs[::skip], 1], flow_gt[ys[::skip], xs[::skip], 0], 
                        alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

            if flownet_model:
                flow_im1 = flow_pr1.squeeze().permute(1, 2, 0).cpu().numpy()
                flow_im1[mask, :] = 0
                
                skip = 15
                ax[4].set_title(f"Flownet EPE: {epe1[val].mean().numpy():0.3f}")
                ax[4].imshow(np.zeros((h, w)), alpha=0.5)
                ys, xs, _ = np.where(flow_im1 != 0)
                ax[4].quiver(xs[::skip], ys[::skip],
                            flow_im1[ys[::skip], xs[::skip], 1], flow_im1[ys[::skip], xs[::skip], 0], 
                            alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)

            plt.tight_layout()
            plt.show()

    epe = np.mean(epe_list)
    print("Validation Towel EPE: %f" % epe)
    if flownet_model:
        epe1 = np.mean(epe1_list)
        print("Flownet Towel EPE: %f" % epe1)
    return {'towel': epe}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--flownet_model', default='')
    parser.add_argument('--iters', type=int, default=12)
    args = parser.parse_args()

    epes = []
    model = torch.nn.DataParallel(RAFT(args))

    if False:
        for i in range(5000, 300001, 5000):
            model_path = f'/home/exx/projects/RAFT/checkpoints/{i}_raft-towel-fixuv.pth'
            model.load_state_dict(torch.load(model_path))

            model.cuda()
            model.eval()

            # create_sintel_submission(model.module, warm_start=True)
            # create_kitti_submission(model.module)

            with open('config.yaml') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

            seed = cfg['seed']
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            # Towel related parameters
            camera_params = {'default_camera': { 'pos': np.array([-0.0, 0.65, 0.0]),
            # camera_params = {'default_camera': { 'pos': np.array([-0.0, 0.45, 0.0]),
                            'angle': np.array([0, -np.pi/2., 0.]),
                            'width': 720,
                            'height': 720}}

            # Train/val split needs to happen outside dataset class
            # to avoid random nobs switch from using val images
            datapath = f'{cfg["basepath"]}/{cfg["valname"]}'
            fs = sorted([int(fn.split('_')[0])
                        for fn in os.listdir(f'{datapath}/rendered_images') 
                        if 'before' in fn])
            random.shuffle(fs)
            # train_fs = fs[:int(len(fs)*0.8)]
            # val_fs = fs[int(len(fs)*0.8):]
            val_fs = fs[:1000]

            with torch.no_grad():
                if args.dataset == 'chairs':
                    validate_chairs(model.module)

                elif args.dataset == 'sintel':
                    validate_sintel(model.module)

                elif args.dataset == 'kitti':
                    validate_kitti(model.module)

                elif args.dataset == 'towel':
                    d = validate_towel(model.module, cfg, val_fs, camera_params, plot=args.plot, flownet_model=args.flownet_model, iters=args.iters)
            epes.append((i, d['towel']))
            np.save("epes.npy", epes)
            time.sleep(5)


    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with open('config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    torch.manual_seed(1234)
    np.random.seed(1234)

    # Towel related parameters
    # camera_params = {'default_camera': { 'pos': np.array([-0.0, 0.45, 0.0]),
    camera_params = {'default_camera': { 'pos': np.array([-0.0, 0.65, 0.0]),
                     'angle': np.array([0, -np.pi/2., 0.]),
                     'width': 720,
                     'height': 720}}

    # Train/val split needs to happen outside dataset class
    # to avoid random nobs switch from using val images
    datapath = f'{cfg["basepath"]}/{cfg["valname"]}'
    fs = sorted([int(fn.split('_')[0])
                for fn in os.listdir(f'{datapath}/rendered_images') 
                if 'before' in fn])
    random.shuffle(fs)
    # train_fs = fs[:int(len(fs)*0.8)]
    # val_fs = fs[int(len(fs)*0.8):]
    val_fs = fs[:1000]
    print(len(val_fs))

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module)

        elif args.dataset == 'sintel':
            validate_sintel(model.module)

        elif args.dataset == 'kitti':
            validate_kitti(model.module)

        elif args.dataset == 'towel':
            validate_towel(model.module, cfg, val_fs, camera_params, plot=args.plot, flownet_model=args.flownet_model, iters=args.iters)
        
        elif args.dataset == 'toweltest':
            validate_towel(model.module, cfg, val_fs, camera_params, plot=args.plot, flownet_model=args.flownet_model, dataset=args.dataset, iters=args.iters)


