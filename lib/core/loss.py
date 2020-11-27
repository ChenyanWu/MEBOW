from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Function
from utils.debug import print_variable
import numpy as np


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)




def _tranpose_and_gather_scalar(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, 1)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


def reg_l1_loss(pred, target ,mask, has_3d_label):
    pred = torch.squeeze(pred)
    target = torch.squeeze(target)
    loss = torch.abs(pred - target) * mask
    loss = loss.sum()
    num = mask.float().sum()
    loss = loss / (num + 1e-4)
    return loss

class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
    def forward(self, output, mask, ind, target, has_3d_label):
        # pay attention that all the variables are depth
        pred = _tranpose_and_gather_scalar(output, ind)
        loss = reg_l1_loss(pred, target ,mask, has_3d_label)
        return loss

class hoe_diff_loss(nn.Module):
    def __init__(self):
        super(hoe_diff_loss, self).__init__()
        # self.softmax_layer = nn.Softmax2d()
        self.compute_norm = nn.L1Loss()

    def forward(self, plane_output, depth_output, hoe_output_val, ind, has_hoe_label):
        # get the width value using intergal
        # plane_output_softmax = self.softmax_layer(plane_output)
        H, W = plane_output.shape[2:]
        plane_output_softmax = plane_output.reshape(plane_output.shape[0], plane_output.shape[1], -1)
        part_sum = plane_output_softmax.sum(2, keepdim=True)
        plane_output_softmax = plane_output_softmax / part_sum
        plane_output_softmax = plane_output_softmax.reshape(plane_output_softmax.shape[0], plane_output_softmax.shape[1], H, W)

        interg_H = torch.sum(plane_output_softmax, 2, keepdim=False)
        W_value = interg_H * torch.arange(1, interg_H.shape[-1]+1).type(torch.cuda.FloatTensor)
        W_value = torch.sum(W_value, 2)
        W_value /= min(H, W)

        # get the depth value
        pred_h = _tranpose_and_gather_scalar(depth_output, ind)
        pred_h = torch.squeeze(pred_h)

        # compute hoe of 3D pose coordinates
        # hoe_from_3d = torch.atan2(pred_h[:, 14] - pred_h[:, 11],
        #                           W_value[:, 14] - W_value[:, 11]) / 3.1415926 * 180 / 5
        hoe_from_3d = torch.atan2(pred_h[:, 14] - pred_h[:, 11],
                                  W_value[:, 14] - W_value[:, 11])
        sin_angle = torch.sin(hoe_from_3d)
        cos_angle = torch.cos(hoe_from_3d)

        sin_gt = torch.sin(hoe_output_val)
        cos_gt = torch.cos(hoe_output_val)
        loss = ((sin_angle - sin_gt) ** 2 + (cos_angle - cos_gt) ** 2)
        loss = loss * has_hoe_label
        loss = loss.sum() / (has_hoe_label.sum() + 1e-4)
        # compute the distance
        # loss = (36 - torch.abs(torch.abs(hoe_output_val - hoe_from_3d) - 36)).mean()
        # loss = self.compute_norm(hoe_from_3d, hoe_output_val)
        return loss

class Bone_loss(nn.Module):
    def __init__(self):
        super(Bone_loss, self).__init__()
    def forward(self, output, mask, ind, target, gt_2d):
        pred = _tranpose_and_gather_scalar(output, ind)
        bone_func = VarLoss(1)
        loss = bone_func(pred, target, mask, gt_2d)
        return loss


class VarLoss(Function):
    def __init__(self, var_weight):
        super(VarLoss, self).__init__()
        self.var_weight = var_weight
        # self.skeleton_idx = [[[0, 1], [1, 2],
        #                       [3, 4], [4, 5]],
        #                      [[10, 11], [11, 12],
        #                       [13, 14], [14, 15]],
        #                      [[2, 6], [3, 6]],
        #                      [[12, 8], [13, 8]]]
        # [0, 1, 2, 3, 4, 5, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15]
        # [3, 2, 1, 4, 5, 6, 0, 7, 8, 10, 16, 15, 14, 11, 12, 13]
        self.skeleton_idx = [[[3, 2], [2, 1],
                              [4, 5], [5, 6]],
                             [[16, 15], [15, 14],
                              [11, 12], [12, 13]],
                             [[1, 0], [4, 0]],
                             [[14, 8], [11, 8]]]
        self.skeleton_weight = [[1.0085885098415446, 1,
                                 1, 1.0085885098415446],
                                [1.1375361376887123, 1,
                                 1, 1.1375361376887123],
                                [1, 1],
                                [1, 1]]

    def forward(self, input, visible, mask, gt_2d):
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        batch_size = input.size(0)
        output = torch.cuda.FloatTensor(1) * 0
        for t in range(batch_size):
            if mask[t].sum() == 0:  # mask is the mask for supervised depth
                # xy[t] = 2.0 * xy[t] / ref.outputRes - 1
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                                    (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    loss = 0
                    for j in range(N):
                        if l[j] > 0:
                            loss += (l[j] - E) ** 2 / 2. / num
                    output += loss
        output = self.var_weight * output / batch_size
        self.save_for_backward(input, visible, mask, gt_2d)
        # output = output.cuda(self.device, non_blocking=True)
        return output

    def backward(self, grad_output):
        input, visible, mask, gt_2d = self.saved_tensors
        xy = gt_2d.view(gt_2d.size(0), -1, 2)
        grad_input = torch.zeros(input.size()).type(torch.cuda.FloatTensor)
        batch_size = input.size(0)
        for t in range(batch_size):
            if mask[t].sum() == 0:  # mask is the mask for supervised depth
                for g in range(len(self.skeleton_idx)):
                    E, num = 0, 0
                    N = len(self.skeleton_idx[g])
                    l = np.zeros(N)
                    for j in range(N):
                        id1, id2 = self.skeleton_idx[g][j]
                        if visible[t, id1] > 0.5 and visible[t, id2] > 0.5:
                            l[j] = (((xy[t, id1] - xy[t, id2]) ** 2).sum() + \
                                    (input[t, id1] - input[t, id2]) ** 2) ** 0.5
                            l[j] = l[j] * self.skeleton_weight[g][j]
                            num += 1
                            E += l[j]
                    if num < 0.5:
                        E = 0
                    else:
                        E = E / num
                    for j in range(N):
                        if l[j] > 0:
                            id1, id2 = self.skeleton_idx[g][j]
                            grad_input[t][id1] += self.var_weight * \
                                                  self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                                                  / l[j] * (input[t, id1] - input[t, id2]) / batch_size
                            grad_input[t][id2] += self.var_weight * \
                                                  self.skeleton_weight[g][j] ** 2 / num * (l[j] - E) \
                                                  / l[j] * (input[t, id2] - input[t, id1]) / batch_size
        return grad_input, None, None, None

