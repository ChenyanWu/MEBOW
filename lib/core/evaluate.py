from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from core.inference import get_max_preds

def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def continous_comp_deg_error(output, gt_degree):
    result = 0
    index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    for i in range(len(index_degree)):
        diff = abs(index_degree[i]*5 - gt_degree[i])
        diff = min(diff, 360 - diff)
        result += diff
        if diff <= 45:
            poor_45 += 1
            if diff <= 30:
                poor += 1
                if diff <= 22.5:
                    poor_225 += 1
                    if diff <= 15:
                        mid += 1
                        if diff <= 5:
                            excellent += 1
    return result/len(output), excellent, mid, poor_225, poor, poor_45, gt_degree, index_degree*5, len(output)

def comp_deg_error(output, degree):
    result = 0
    degree = degree.argmax(axis = 1)
    # index_degree is the prediction
    index_degree = output.argmax(axis = 1)
    excellent = 0
    mid = 0
    poor_225 = 0
    poor = 0
    poor_45 = 0
    for i in range(len(index_degree)):
        diff = abs(index_degree[i] - degree[i]) * 5
        diff = min(diff, 360 - diff)
        result += diff
        if diff <= 45:
            poor_45 += 1
            if diff <= 30:
                poor += 1
                if diff <= 22.5:
                    poor_225 += 1
                    if diff <= 15:
                        mid += 1
                        if diff <= 5:
                            excellent += 1
    return result/len(output), excellent, mid, poor_225, poor, poor_45,degree*5 ,index_degree * 5, len(output)



# this is for human3.6
def mpjpe(heatmap, depthmap, gt_3d, convert_func):
  preds_3d = get_preds_3d(heatmap, depthmap)
  cnt, pjpe = 0, 0
  for i in range(preds_3d.shape[0]):
    if gt_3d[i].sum() ** 2 > 0:
      cnt += 1
      pred_3d_h36m = convert_func(preds_3d[i])
      err = (((gt_3d[i] - pred_3d_h36m) ** 2).sum(axis=1) ** 0.5).mean()
      pjpe += err
  if cnt > 0:
    pjpe /= cnt
  return pjpe, cnt

def get_preds_3d(heatmap, depthmap):
  output_res = min(heatmap.shape[2], heatmap.shape[3])
  preds = get_preds(heatmap).astype(np.int32)
  preds_3d = np.zeros((preds.shape[0], preds.shape[1], 3), dtype=np.float32)
  for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
      idx = min(j, depthmap.shape[1] - 1)
      pt = preds[i, j]
      preds_3d[i, j, 2] = depthmap[i, idx, pt[1], pt[0]]
      preds_3d[i, j, :2] = 1.0 * preds[i, j] / output_res
    preds_3d[i] = preds_3d[i] - preds_3d[i, 0:1]
  return preds_3d

def get_preds(hm, return_conf=False):
    assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
    h = hm.shape[2]
    w = hm.shape[3]
    hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
    idx = np.argmax(hm, axis=2)

    preds = np.zeros((hm.shape[0], hm.shape[1], 2))
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            preds[i, j, 0], preds[i, j, 1] = idx[i, j] % w, idx[i, j] / w
    if return_conf:
        conf = np.amax(hm, axis=2).reshape(hm.shape[0], hm.shape[1], 1)
        return preds, conf
    else:
        return preds

# this is to  draw images
def draw_orientation(img_np, gt_ori, pred_ori , path, alis=''):
    for idx in range(len(gt_ori)):
        img_tmp = img_np[idx]
        img_tmp = np.transpose(img_tmp, axes=[1, 2, 0])
        img_tmp *= [0.229, 0.224, 0.225]
        img_tmp += [0.485, 0.456, 0.406]
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)

        # then draw the image
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)

        theta_1 = gt_ori[idx]/180 * np.pi + np.pi/2
        theta_2 = pred_ori[idx]/180 * np.pi + np.pi/2
        plt.plot([0, np.cos(theta_1)], [0, np.sin(theta_1)], color="red", linewidth=3)
        plt.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color="blue", linewidth=3)
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        fig.savefig(os.path.join(path, str(idx)+'_'+alis+'.jpg'))
        ori_img = cv2.imread(os.path.join(path, str(idx)+'_'+alis+'.jpg'))

        width = img_tmp.shape[1]
        ori_img = cv2.resize(ori_img, (width, width), interpolation=cv2.INTER_CUBIC)
        img_all = np.concatenate([img_tmp, ori_img],axis=0)
        im = Image.fromarray(img_all)
        im.save(os.path.join(path, str(idx)+'_'+alis+'_raw.jpg'))

def ori_numpy(gt_ori, pred_ori):
    ori_list = []
    for idx in range(len(gt_ori)):
        ori_list.append((gt_ori[idx], pred_ori[idx]))
    return ori_list









