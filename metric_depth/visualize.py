import os
import time
import torch
import argparse
from tqdm import tqdm
from pprint import pprint
import torch.nn.functional as F
from os.path import join, exists
import torchvision.transforms as transforms

from zoedepth.models.builder import build_model
from zoedepth.data.data_mono import DepthDataLoader

from zoedepth.utils.misc import colorize
from zoedepth.utils.depth_utils import depth2xyz
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.easydict import EasyDict as edict
from zoedepth.utils.misc import RunningAverageDict, colors, compute_metrics, count_parameters
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR, HOME_DIR

from easyvolcap.utils.data_utils import export_pts, save_image, to_numpy


def save(depth, preds, image, batch, config):
    # Get and create the full result directory
    depths_dir = join(HOME_DIR, config.result_dir, "DEPTH")
    points_dir = join(HOME_DIR, config.result_dir, "POINT")
    os.makedirs(depths_dir, exist_ok=True)
    os.makedirs(points_dir, exist_ok=True)

    # Get the index of the sample
    name = f"frame{batch['frame_index'].item():04d}_camera{batch['camera_index'].item():04d}"

    # Save the ground truth depth map if available
    if depth is not None:
        save_image(join(depths_dir, f'{name}_gt.jpg'), colorize(depth[0, 0].cpu().numpy(), 0, 10))  # ! BATCH = 1
    # Save the predicted depth map
    save_image(join(depths_dir, f'{name}.jpg'), colorize(preds[0, 0].cpu().numpy(), 0, 10))  # ! BATCH = 1
    # Save the raw rgb image
    save_image(join(depths_dir, f'{name}_rgb.jpg'), image)

    # Save the ground truth point cloud if available
    if depth is not None:
        gxyz = depth2xyz(depth.cpu(), batch['ixt'][:, None], batch['w2c'][:, None])[0, 0]  # (B, 1, 3, H, W) -> (3, H, W)
        export_pts(to_numpy(gxyz.permute(1, 2, 0).reshape(-1, 3)),
                to_numpy(image[0].permute(1, 2, 0).reshape(-1, 3)),
                filename=join(points_dir, f'{name}_gt.ply'))
    # Save the predicted point cloud
    pxyz = depth2xyz(preds.cpu(), batch['ixt'][:, None], batch['w2c'][:, None])[0, 0]  # (B, 1, 3, H, W) -> (3, H, W)
    export_pts(to_numpy(pxyz.permute(1, 2, 0).reshape(-1, 3)),
               to_numpy(image[0].permute(1, 2, 0).reshape(-1, 3)),
               filename=join(points_dir, f'{name}.ply'))


@torch.no_grad()
def visualize(model, test_loader, config):
    # Evaluate on all the test data
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Fetch the image and depth from the sample and move to GPU
        image = batch['image'].cuda()  # (B, 3, H, W)

        # Forward pass
        torch.cuda.synchronize()
        s = time.time()
        preds = model(image, dataset=batch['dataset'][0])['metric_depth']  # (B, 1, H, W)
        # Interpolate the predicted depth to the original size
        preds = F.interpolate(preds, (batch['H'][0].item(), batch['W'][0].item()), mode='bilinear', align_corners=False)  # (B, 1, H, W)
        torch.cuda.synchronize()
        e = time.time()
        print(f"Network time for processing a image of size {batch['H'][0].item()} x {batch['W'][0].item()} is {e-s}")

        # Save image and predicted depth for visualization
        save(None, preds, image, batch, config)

    return None


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    metrics = RunningAverageDict()

    # Evaluate on all the test data
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        # Skip samples with invalid depth
        # What is this? When this circumstance occurs?
        if 'has_valid_depth' in batch:
            if not batch['has_valid_depth']:
                continue

        # Fetch the image and depth from the sample and move to GPU
        image, depth = batch['image'].cuda(), batch['depth'].cuda()  # (B, 3, H, W), (B, 1, H, W)

        # Forward pass
        torch.cuda.synchronize()
        s = time.time()
        preds = model(image, dataset=batch['dataset'][0])['metric_depth']  # (B, 1, Hn, Wn)
        # Interpolate the predicted depth to the original size
        preds = F.interpolate(preds, (batch['H'][0].item(), batch['W'][0].item()), mode='bilinear', align_corners=False)  # (B, 1, H, W)
        torch.cuda.synchronize()
        e = time.time()
        print(f"Network time for processing a image of size {batch['H'][0].item()} x {batch['W'][0].item()} is {e-s}")

        # Save image, ground truth depth, and predicted depth for visualization
        save(depth, preds, image, batch, config)

        # Compute the metrics for this batch
        metrics.update(compute_metrics(depth, preds, config=config))

    # Round the metrics if required
    r = lambda m: round(m, round_precision) if round_vals else m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}

    return metrics

def main(config, visualize_only=False):
    # Build the model, move to GPU and set as eval mode
    model = build_model(config)
    model = model.cuda()
    model.eval()

    # Build the test dataloader
    test_loader = DepthDataLoader(config, 'online_eval').data

    # Evaluate the model
    if not visualize_only:
        metrics = evaluate(model, test_loader, config)
        print(f"{colors.fg.green}")
        print(metrics)
        print(f"{colors.reset}")
        metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    # Only visualize the prediction
    else:
        visualize(model, test_loader, config)
        metrics = None

    return metrics


def eval_model(model_name, pretrained_resource, dataset='easyvolcap_test', visualize_only=False, **kwargs):
    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    pprint(config)

    # Start evaluating or visualizing
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config, visualize_only)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str, required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False, default='easyvolcap_test', help="Dataset to evaluate on")
    parser.add_argument("-v", "--visualize_only", action='store_true', help="Only display the prediction")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset: datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset: datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset: datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset: datasets = args.dataset.split(",")
    else: datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource, dataset=dataset, visualize_only=args.visualize_only, **overwrite_kwargs)
