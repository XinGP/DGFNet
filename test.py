import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeter, AverageMeterForDict


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    test_set, net, loss_fn, _, evaluator = loader.load()

    dl_val = DataLoader(test_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=test_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()

    # begin inference
    preds = {}
    gts = {}
    cities = {}
    probabilities = {}
    for i, data in enumerate(tqdm(dl_val)):
        with torch.no_grad():
            orig = data['ORIG']
            rot = data['ROT']
            orig_tensor = orig[0]
            rot_tensor = rot[0]
            orig_np = orig_tensor.cpu().numpy()
            rot_np = rot_tensor.cpu().numpy()

            data_in = net.pre_process(data)
            out = net(data_in)
            post_out = net.post_process(out)

            results = [x.detach().cpu().numpy() for x in post_out['traj_pred']]
            results_orig = []
            for traj_pred in results:
                trajs_rotated = (traj_pred).dot(rot_np.T)
                trajs_restored = trajs_rotated + orig_np
                results_orig.append(trajs_restored)
            probs = [x.detach().cpu().numpy() for x in post_out['prob_pred']]

        for i, (argo_idx, pred_traj, pred_prob) in enumerate(zip(data['SEQ_ID'], results_orig, probs)):
            preds[argo_idx] = pred_traj.squeeze()
            x = pred_prob.squeeze()
            probabilities[argo_idx] = x
            #probabilities[argo_idx] = softmax(x)
            #print(probabilities[argo_idx])
            for i in range(len(probabilities[argo_idx])):
                if probabilities[argo_idx][i]<0.1:
                    probabilities[argo_idx][0] = probabilities[argo_idx][0]+probabilities[argo_idx][i]
                    probabilities[argo_idx][i]=0
                    
            if probabilities[argo_idx][0]>0.45:
                probabilities[argo_idx][0] = (probabilities[argo_idx][0]
                                              +probabilities[argo_idx][3]
                                              +probabilities[argo_idx][4]
                                              +probabilities[argo_idx][5])
                probabilities[argo_idx][3] = 0
                probabilities[argo_idx][4] = 0
                probabilities[argo_idx][5] = 0
           
            if probabilities[argo_idx][0]>0.6:
                probabilities[argo_idx][0] = 1
                probabilities[argo_idx][1] = 0
                probabilities[argo_idx][2] = 0
                probabilities[argo_idx][3] = 0
                probabilities[argo_idx][4] = 0
                probabilities[argo_idx][5] = 0
                
            cities[argo_idx] = data['CITY_NAME']
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None


    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )

    from argoverse.evaluation.competition_util import generate_forecasting_h5
    generate_forecasting_h5(preds, f"/home/wuyou/simpl1/6.23/submit.h5",probabilities=probabilities)  # this might take awhile
    import ipdb;ipdb.set_trace()

def reconstruct_trajectories(results):
    trajectories = []
    for traj in results:
        # 初始位置
        x_prev, y_prev = 0, 0
        reconstructed_traj = []
        for step in traj:
            # 计算当前步的坐标
            x = x_prev + step[0]
            y = y_prev + step[1]
            # 更新前一个位置
            x_prev, y_prev = x, y
            # 添加到重构的轨迹中
            reconstructed_traj.append([x, y])
        # 将重构的轨迹添加到轨迹列表中
        trajectories.append(reconstructed_traj)
    return trajectories

if __name__ == "__main__":
    main()
