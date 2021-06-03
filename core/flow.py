import numpy as np
import matplotlib.pyplot as plt

import cv2
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init

import math

class GTFlow(object):
    def __init__(self, normalize=False):
        # self.norm = 200 if normalize else 1
        pass

    def get_image(self, uv_o, uv_g_f, mask=None, depth_o=None, depth_n=None):
        # Compute uv diff
        uv_diff = np.rint((uv_g_f - uv_o)/719*199)

        if False:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(depth_o)
            ax[1].imshow(depth_n)
            plt.show()
        # im = np.zeros((200, 200))
        # ax[2].imshow(im)
        # ax[2].scatter(uv_om[:, 1], uv_om[:, 0], c='r', s=0.1)
        # plt.show()
        # import IPython; IPython.embed()

        # mask uv_diff for the values that are not on the mask
        non_nan_idxs = ~np.isnan(uv_diff).any(axis=1)
        uv_diff_nonan = uv_diff[non_nan_idxs]
        uv_o_nonan = np.rint(uv_o[non_nan_idxs]/719*199).astype(int)
        # print(f'Any nan in uv_g: {np.any(np.isnan(uv_g_f))}')

        # Make diff image
        # uv_diff_nonan = uv_diff_nonan / self.norm # normalize
        im_diff_sp = np.zeros((200, 200, 2))
        try:
            im_diff_sp[uv_o_nonan[:, 0], uv_o_nonan[:, 1], :] = uv_diff_nonan
        except Exception as e:
            print(e)
            import IPython; IPython.embed()

        if False: # visualize
            s = 20
            # uv_diff_nonan = uv_diff_nonan * self.norm # unnormalize
            # rgb_o = cv2.resize(rgb_o, (200, 200))
            im = np.zeros((200, 200))
            fig, ax = plt.subplots(1, 5, figsize=(16, 8))
            ax[0].set_title('occluded particles')
            uv_o_occl = np.rint(uv_o[np.isnan(uv_diff).any(axis=1)]/719*199).astype(int) # Doesn't work, need original uv_o
            ax[0].scatter(uv_o_occl[:, 1], uv_o_occl[:, 0], c='r', s=0.1)
            # ax[0].imshow(rgb_o)
            ax[0].imshow(im)
            ax[1].set_title('visible particles')
            # ax[1].imshow(rgb_o)
            ax[1].imshow(im)
            ax[1].scatter(uv_o_nonan[:, 1], uv_o_nonan[:, 0], c='b', s=0.1)
            ax[2].set_title('sparse flow plot')
            ax[2].imshow(im)
            ax[2].quiver(uv_o_nonan[::, 1], uv_o_nonan[::, 0], 
                        uv_diff_nonan[::, 1], uv_diff_nonan[::, 0], 
                        alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)
            ax[3].set_title('sparse flow plot (downsampled)')
            ax[3].imshow(im)
            ax[3].quiver(uv_o_nonan[::s, 1], uv_o_nonan[::s, 0], 
                        uv_diff_nonan[::s, 1], uv_diff_nonan[::s, 0], 
                        alpha=0.8, color='white', angles='xy', scale_units='xy', scale=1)
            ax[4].set_title('goal image')
            # ax[4].imshow(rgb_g)
            ax[4].imshow(im)
            uv_g = np.rint(uv_g_f/719*199).astype(int)
            ax[4].scatter(uv_g[:, 1], uv_g[:, 0], c='r', s=0.1)
            plt.show()

        return im_diff_sp

def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

def get_rotation_matrix(angle, axis):
	axis = axis / np.linalg.norm(axis)
	s = np.sin(angle)
	c = np.cos(angle)

	m = np.zeros((4, 4))

	m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
	# m[0][1] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
	m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
	# m[0][2] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
	m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
	m[0][3] = 0.0

	# m[1][0] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
	m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
	m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
	# m[1][2] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
	m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
	m[1][3] = 0.0

	# m[2][0] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
	m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
	# m[2][1] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
	m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
	m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
	m[2][3] = 0.0

	m[3][0] = 0.0
	m[3][1] = 0.0
	m[3][2] = 0.0
	m[3][3] = 1.0

	return m

def uv_to_world_pos(camera_params, depth, u, v, particle_radius=0.0075, on_table=False):
    # height, width, _ = rgb.shape
    height, width = depth.shape
    K = intrinsic_from_fov(height, width, 45) # the fov is 90 degrees

    # from cam coord to world coord
    cam_x, cam_y, cam_z = camera_params['default_camera']['pos'][0], camera_params['default_camera']['pos'][1], camera_params['default_camera']['pos'][2]
    cam_x_angle, cam_y_angle, cam_z_angle = camera_params['default_camera']['angle'][0], camera_params['default_camera']['angle'][1], camera_params['default_camera']['angle'][2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0]) 
    # matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [np.cos(cam_x_angle), 0, np.sin(cam_x_angle)])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z
    matrix = np.linalg.inv(rotation_matrix @ translation_matrix)

    cam_coord = np.ones(4)
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    z = depth[int(np.rint(u)), int(np.rint(v))]
    if on_table or z == 0:
        vec = ((v - x0) / fx, (u - y0) / fy)
        z = (particle_radius - matrix[1, 3]) / (vec[0] * matrix[1, 0] + vec[1] * matrix[1, 1] + matrix[1, 2])
    else:
        # adjust for particle radius from depth image
        z -= particle_radius
        
    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    
    cam_coord[:3] = (x, y, z)
    #print(cam_coord)
    #print(x,y,z)
    world_coord = matrix @ cam_coord

    return world_coord

def remove_dups(camera_params, knots, coords, depth, rgb=None, zthresh=0.001):
    knots = deepcopy(knots)
    if depth.shape[0] < camera_params['default_camera']['height']:
        print('Warning: resizing depth')
        depth = cv2.resize(depth, (camera_params['default_camera']['height'], camera_params['default_camera']['width']))

    unoccluded_knots = []
    occluded_knots = []
    for i, uv in enumerate(knots):
        u_f, v_f = uv[0], uv[1]
        if np.isnan(u_f) or np.isnan(v_f):
            continue
        u, v = int(np.rint(u_f)), int(np.rint(v_f))

        if u < 0 or v < 0 or u >= depth.shape[0] or v >= depth.shape[1]:
            # pixel is outside of image bounds
            knots[i] = [float('NaN'), float('NaN')]
            continue
        
        d = depth[u, v]

        # Get depth into world coordinates
        proj_coords = uv_to_world_pos(camera_params, depth, u_f, v_f, particle_radius=0, on_table=False)[0:3]
        z_diff = proj_coords[1] - coords[i][1]

        # Check is well projected xyz point
        if z_diff > zthresh:
            # invalidate u, v and continue
            occluded_knots.append(deepcopy(knots[i]))
            knots[i] = [float('NaN'), float('NaN')]
            continue

        unoccluded_knots.append(deepcopy(knots[i]))
    
    # print("unoccluded knots: ", len(unoccluded_knots))

    if False: # debug visualization
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # for i, (u, v) in enumerate(knots):
        #     c = 'r' if np.isnan(u) or np.isnan(v) else 'b'
        #     ax.scatter(coords[i, 0], coords[i, 2], coords[i, 1], s=1, c=c)
        # plt.show()

        fig, ax = plt.subplots(1, 3, dpi=200)
        ax[0].set_title('depth')
        ax[0].imshow(depth)
        ax[1].set_title('occluded points\nin red')
        ax[1].imshow(depth)
        if occluded_knots != []:
            occluded_knots = np.array(occluded_knots)
            ax[1].scatter(occluded_knots[:, 1], occluded_knots[:, 0], marker='.', s=1, c='r', alpha=0.4)
        ax[2].imshow(depth)
        ax[2].set_title('unoccluded points\nin blue')
        unoccluded_knots = np.array(unoccluded_knots)
        ax[2].scatter(unoccluded_knots[:, 1], unoccluded_knots[:, 0], marker='.', s=1, alpha=0.4)
        plt.show()
        
    return knots

import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

# from .submodules import *
# 'Parameter count : 38,676,504 '

import torch.nn as nn
import torch
import numpy as np 
# from torchsummary import summary

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, ksize=3):
    return nn.Sequential(
        # nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=ksize, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

class FlowNetSmall(nn.Module):
    def __init__(self, input_channels = 12, batchNorm=True):
        super(FlowNetSmall,self).__init__()

        fs = [8, 16, 32, 64, 128] # filter sizes
        # fs = [16, 32, 64, 128, 256] # filter sizes
        # fs = [64, 128, 256, 512, 1024] # filter sizes
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm, input_channels, fs[0], kernel_size=7, stride=2) # 384 -> (384 - 7 + 2*3)/2 + 1 = 377
        self.conv2   = conv(self.batchNorm, fs[0], fs[1], kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, fs[1], fs[2], kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, fs[2], fs[2])
        self.conv4   = conv(self.batchNorm, fs[2], fs[3], stride=2)
        self.conv4_1 = conv(self.batchNorm, fs[3], fs[3])
        self.conv5   = conv(self.batchNorm, fs[3], fs[3], stride=2)
        self.conv5_1 = conv(self.batchNorm, fs[3], fs[3])
        self.conv6   = conv(self.batchNorm, fs[3], fs[4], stride=2)
        self.conv6_1 = conv(self.batchNorm, fs[4], fs[4])

        self.deconv5 = deconv(fs[4],fs[3])
        self.deconv4 = deconv(fs[3]+fs[3]+2,fs[2])
        self.deconv3 = deconv(fs[3]+fs[2]+2,fs[1])
        self.deconv2 = deconv(fs[2]+fs[1]+2,fs[0], ksize=4)
        # self.deconv5 = deconv(1024,512)
        # self.deconv4 = deconv(1026,256)
        # self.deconv3 = deconv(770,128)
        # self.deconv2 = deconv(386,64, ksize=4)

        self.predict_flow6 = predict_flow(fs[4])
        self.predict_flow5 = predict_flow(fs[3]+fs[3]+2)
        self.predict_flow4 = predict_flow(fs[3]+fs[2]+2)
        self.predict_flow3 = predict_flow(fs[2]+fs[1]+2)
        self.predict_flow2 = predict_flow(fs[1]+fs[0]+2)

        # self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False) # (H_in-1)*stride - 2*padding + (kernel-1) + 1
        # self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 3, 2, 1, bias=False) # (H_in-1)*stride - 2*padding + (kernel-1) + 1
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 3, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 3, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
                # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        # print(out_conv5.size(), out_deconv5.size(), flow6_up.size())      

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)

        # print(out_conv4.size(), out_deconv4.size(), flow5_up.size())
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)

        # print(out_conv3.size(), out_deconv3.size(), flow4_up.size())

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        # print(out_conv2.size(), out_deconv2.size(), flow3_up.size())

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        # print(flow2.size())

        # if self.training:
            # return flow2,flow3,flow4,flow5,flow6
        # else:

        out = self.upsample1(flow2)

        return out

# if __name__ == '__main__':
#     f = FlowNetSmall(input_channels=2).cuda()
#     print(summary(f, [(2, 200, 200)]))
    # print(summary(f, [(2, 384, 384)]))

# def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias = True):
#     if batchNorm:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
#             nn.BatchNorm2d(out_planes),
#         )
#     else:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
#         )

# class tofp16(nn.Module):
#     def __init__(self):
#         super(tofp16, self).__init__()

#     def forward(self, input):
#         return input.half()

# class tofp32(nn.Module):
#     def __init__(self):
#         super(tofp32, self).__init__()

#     def forward(self, input):
#         return input.float()

# def init_deconv_bilinear(weight):
#     f_shape = weight.size()
#     heigh, width = f_shape[-2], f_shape[-1]
#     f = np.ceil(width/2.0)
#     c = (2 * f - 1 - f % 2) / (2.0 * f)
#     bilinear = np.zeros([heigh, width])
#     for x in range(width):
#         for y in range(heigh):
#             value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
#             bilinear[x, y] = value
#     weight.data.fill_(0.)
#     for i in range(f_shape[0]):
#         for j in range(f_shape[1]):
#             weight.data[i,j,:,:] = torch.from_numpy(bilinear)

# def save_grad(grads, name):
#     def hook(grad):
#         grads[name] = grad
#     return hook

# class FlowNetS(nn.Module):
#     def __init__(self, input_channels = 12, batchNorm=True):
#         super(FlowNetS,self).__init__()

#         self.batchNorm = batchNorm
#         self.conv1   = conv(self.batchNorm,  input_channels,   64, kernel_size=7, stride=2)
#         self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
#         self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
#         self.conv3_1 = conv(self.batchNorm, 256,  256)
#         self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
#         self.conv4_1 = conv(self.batchNorm, 512,  512)
#         self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
#         self.conv5_1 = conv(self.batchNorm, 512,  512)
#         self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
#         self.conv6_1 = conv(self.batchNorm,1024, 1024)

#         self.deconv5 = deconv(1024,512)
#         self.deconv4 = deconv(1026,256)
#         self.deconv3 = deconv(770,128)
#         self.deconv2 = deconv(386,64)

#         self.predict_flow6 = predict_flow(1024)
#         self.predict_flow5 = predict_flow(1026)
#         self.predict_flow4 = predict_flow(770)
#         self.predict_flow3 = predict_flow(386)
#         self.predict_flow2 = predict_flow(194)

#         self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
#         self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)

#             if isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     init.uniform_(m.bias)
#                 init.xavier_uniform_(m.weight)
#                 # init_deconv_bilinear(m.weight)
#         self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

#     def forward(self, x):
#         out_conv1 = self.conv1(x)

#         out_conv2 = self.conv2(out_conv1)
#         out_conv3 = self.conv3_1(self.conv3(out_conv2))
#         out_conv4 = self.conv4_1(self.conv4(out_conv3))
#         out_conv5 = self.conv5_1(self.conv5(out_conv4))
#         out_conv6 = self.conv6_1(self.conv6(out_conv5))

#         flow6       = self.predict_flow6(out_conv6)
#         flow6_up    = self.upsampled_flow6_to_5(flow6)
#         out_deconv5 = self.deconv5(out_conv6)
        
#         print(out_conv5.size(), out_deconv5.size(), flow6_up.size())

#         # import IPython; IPython.embed()
#         concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
#         flow5       = self.predict_flow5(concat5)
#         flow5_up    = self.upsampled_flow5_to_4(flow5)
#         out_deconv4 = self.deconv4(concat5)
        
#         concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
#         flow4       = self.predict_flow4(concat4)
#         flow4_up    = self.upsampled_flow4_to_3(flow4)
#         out_deconv3 = self.deconv3(concat4)
        
#         concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
#         flow3       = self.predict_flow3(concat3)
#         flow3_up    = self.upsampled_flow3_to_2(flow3)
#         out_deconv2 = self.deconv2(concat3)

#         concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
#         flow2 = self.predict_flow2(concat2)

#         if self.training:
#             return flow2,flow3,flow4,flow5,flow6
#         else:
#             return flow2,