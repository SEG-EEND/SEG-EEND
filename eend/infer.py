#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from backend.models import (
    average_checkpoints,
    get_model,
)
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu
from os.path import join
from pathlib import Path
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from train import _convert
from types import SimpleNamespace
from typing import TextIO
import logging
import numpy as np
import os
import random
import torch
import yamlargparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_infer_dataloader(args: SimpleNamespace) -> DataLoader:
    infer_set = KaldiDiarizationDataset(
        args.infer_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,
        subsampling=args.subsampling,
        use_last_samples=True,
        min_length=0,
    )
    infer_loader = DataLoader(
        infer_set,
        batch_size=1,
        collate_fn=_convert,
        num_workers=0,
        shuffle=False,
        worker_init_fn=_init_fn,
    )

    Y, _, _ = infer_set.__getitem__(0)
    assert Y.shape[1] == \
        (args.feature_dim * (1 + 2 * args.context_size)), \
        f"Expected feature dimensionality of \
        {args.feature_dim} but {Y.shape[1]} found."
    return infer_loader


def hard_labels_to_rttm(
    labels: np.ndarray,
    id_file: str,
    rttm_file: TextIO,
    frameshift: float = 10
) -> None:
    """
    Transform NfxNs matrix to an rttm file
    Nf is the number of frames
    Ns is the number of speakers
    The frameshift (in ms) determines how to interpret the frames in the array
    """
    if len(labels.shape) > 1:
        # Remove speakers that do not speak
        non_empty_speakers = np.where(labels.sum(axis=0) != 0)[0]
        labels = labels[:, non_empty_speakers]

    # Add 0's before first frame to use diff
    if len(labels.shape) > 1:
        labels = np.vstack([np.zeros((1, labels.shape[1])), labels])
    else:
        labels = np.vstack([np.zeros(1), labels])
    d = np.diff(labels, axis=0)

    spk_list = []
    ini_list = []
    end_list = []
    if len(labels.shape) > 1:
        n_spks = labels.shape[1]
    else:
        n_spks = 1
    for spk in range(n_spks):
        if n_spks > 1:
            ini_indices = np.where(d[:, spk] == 1)[0]
            end_indices = np.where(d[:, spk] == -1)[0]
        else:
            ini_indices = np.where(d[:] == 1)[0]
            end_indices = np.where(d[:] == -1)[0]
        # Add final mark if needed
        if len(ini_indices) == len(end_indices) + 1:
            end_indices = np.hstack([
                end_indices,
                labels.shape[0] - 1])
        assert len(ini_indices) == len(end_indices), \
            "Quantities of start and end of segments mismatch. \
            Are speaker labels correct?"
        n_segments = len(ini_indices)
        for index in range(n_segments):
            spk_list.append(spk)
            ini_list.append(ini_indices[index])
            end_list.append(end_indices[index])
    for ini, end, spk in sorted(zip(ini_list, end_list, spk_list)):
        rttm_file.write(
            f"SPEAKER {id_file} 1 " +
            f"{round(ini * frameshift / 1000, 3)} " +
            f"{round((end - ini) * frameshift / 1000, 3)} " +
            f"<NA> <NA> spk{spk} <NA> <NA>\n")


def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
# Padding function (modified)
def pad_to_max_length(arr, max_T):
    """ Add padding to match different length data to max_T """
    arr = np.array(arr)  # Convert to a NumPy array if input is a list

    # Apply squeeze() if the shape is (1, T, C)
    arr = arr.squeeze(axis=0) if arr.shape[0] == 1 else arr

    # Check shape again
    if arr.ndim != 2:
        raise ValueError(f"❌ Unexpected shape {arr.shape}, expected (T, C)")

    # Apply padding
    T, C = arr.shape
    pad_width = [(0, max_T - T), (0, 0)]  # Add padding only along the time axis (T)
    return np.pad(arr, pad_width, mode='constant', constant_values=0)



def postprocess_output(
    probabilities: torch.Tensor,
    subsampling: int,
    threshold: float,
    median_window_length: int
) -> np.ndarray:
    thresholded = probabilities > threshold
    filtered = np.zeros(thresholded.shape)

    for spk in range(filtered.shape[1]):
        # Move CUDA tensor to CPU and convert it to a NumPy array
        filtered[:, spk] = medfilt(thresholded[:, spk].cpu().numpy(), kernel_size=median_window_length)

    probs_extended = np.repeat(filtered, subsampling, axis=0)
    return probs_extended


def parse_arguments() -> SimpleNamespace:
    parser = yamlargparse.ArgumentParser(description='EEND inference')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--epochs', type=str,
                        help='epochs to average separated by commas \
                        or - for intervals.')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--infer-data-dir', help='inference data directory.')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--median-window-length', default=11, type=int)
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--models-path', type=str,
                        help='directory with model(s) to evaluate')
    parser.add_argument('--num-frames', default=-1, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int)
    parser.add_argument('--rttms-dir', type=str,
                        help='output directory for rttm files.')
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument('--attractor-loss-ratio', default=1.0,
                                type=float, help='weighting parameter')
    attractor_args.add_argument('--attractor-encoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--attractor-decoder-dropout',
                                default=0.1, type=float)
    attractor_args.add_argument('--estimate-spk-qty', default=-1, type=int)
    attractor_args.add_argument('--estimate-spk-qty-thr',
                                default=-1, type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', default=False, type=bool,
        help='If True, avoid backpropagation on attractor loss')
    
    # Additional arguments for SSCD and segmentation
    parser.add_argument('--scd_loss_ratio', default=1.0, type=float, help='State Change Detector loss weight')
    parser.add_argument('--seg_PIT_loss_ratio', default=1.0, type=float, help='Segment PIT loss weight')
    parser.add_argument('--state_change_detector_dropout', default=0.1, type=float, help='Dropout for SSCD')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_arguments()

    # Set seed for reproducibility
    setup_seed(args.seed)
    logging.info(args)

    # Optimize GPU settings
    args.device = torch.device("cuda" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {args.device}")

    # Set up the inference data loader
    infer_loader = get_infer_dataloader(args)

    # Check `estimate_spk_qty` values
    if args.estimate_spk_qty == -1 and args.estimate_spk_qty_thr == -1:
        raise ValueError("Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' must be defined.")

    # Load model and remove `DataParallel` if applied
    model = get_model(args)
    model = average_checkpoints(args.device, model, args.models_path, args.epochs)
    model.eval()
    model.to(args.device)

    # Remove DataParallel wrapper if applied
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Set up output directory (converted to Path object)
    out_dir = Path(args.rttms_dir) / f"epochs{args.epochs}"
    if args.estimate_spk_qty != -1:
        out_dir /= f"spk_qty{args.estimate_spk_qty}"
    else:
        out_dir /= f"spk_qty_thr{args.estimate_spk_qty_thr}"

    out_dir /= Path(f"detection_thr{args.threshold}") / Path(f"median{args.median_window_length}") / "rttms"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Inference Loop
    for batch in infer_loader:
        input_tensor = torch.stack(batch['xs']).to(args.device)
        name = batch['names'][0]

        with torch.no_grad():
            if isinstance(model, torch.nn.DataParallel):
                y_pred = model.module.estimate_sequential(input_tensor, args)[0]
            else:
                y_pred = model.estimate_sequential(input_tensor, args)[0]

        post_y = postprocess_output(y_pred, args.subsampling, args.threshold, args.median_window_length)
                     
        rttm_filename = out_dir / f"{name}.rttm"
        with open(rttm_filename, 'w') as rttm_file:
            hard_labels_to_rttm(post_y, name, rttm_file)