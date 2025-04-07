#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (authors: Federico Landini)
# Licensed under the MIT license.


from backend.models import (
    average_checkpoints,
    get_model,
    load_checkpoint,
    pad_labels,
    pad_sequence,
    save_checkpoint,
)
from backend.updater import setup_optimizer, get_rate
from common_utils.diarization_dataset import KaldiDiarizationDataset
from common_utils.gpu_utils import use_single_gpu, setup_model_for_gpus
from common_utils.metrics import (
    calculate_metrics,
    new_metrics,
    reset_metrics,
    update_metrics,
)
from torch.utils.data import DataLoader, DistributedSampler

# Multi-GPU support
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# For training monitoring
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
import numpy as np
import os
import random
import torch
import logging
import yamlargparse
import re

# Set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
def _init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Convert batch format
def _convert(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]]
) -> Dict[str, Any]: #def dummy():
    return {'xs': [x for x, _, _ in batch],
            'ts': [t for _, t, _ in batch],
            'names': [r for _, _, r in batch]}

# Compute loss and evaluation metrics
def compute_loss_and_metrics(
    model: torch.nn.Module,
    labels: torch.Tensor,
    input: torch.Tensor,
    n_speakers: List[int],
    acum_metrics: Dict[str, float],
    vad_loss_weight: float,
    detach_attractor_loss: bool,
    model_type: str,
) -> Tuple[torch.Tensor, Dict[str, float]]: #def dummy():
    # Ensure compatibility with DataParallel
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    # Move inputs and labels to the same device
    labels = labels.to(input.device)
    input = input.to(input.device)
    
    if model_type == 'TransformerEDA':
        y_pred, attractor_loss = model(input, labels, n_speakers, args)
        loss, standard_loss = model.get_loss(
            y_pred, labels, n_speakers, attractor_loss, vad_loss_weight,
            detach_attractor_loss)
    elif model_type == 'TransformerSCDEDA':
        loss, standard_loss, attractor_loss, scd_loss, seg_PIT_loss, y_pred = model.get_loss(
            input, labels, n_speakers, detach_attractor_loss)
    
    metrics = calculate_metrics(
        labels.detach(), y_pred.detach(), threshold=0.5)
    acum_metrics = update_metrics(acum_metrics, metrics)
    
    # Initialize loss tracking
    if model_type == 'TransformerEDA':
        acum_metrics['loss'] = acum_metrics.get('loss', 0) + loss.item()
        acum_metrics['loss_standard'] = acum_metrics.get('loss_standard', 0) + (standard_loss.item() if standard_loss is not None else 0)
        acum_metrics['loss_attractor'] = acum_metrics.get('loss_attractor', 0) + (attractor_loss.item() if attractor_loss is not None else 0)
        
    elif model_type == 'TransformerSCDEDA':
        acum_metrics['loss'] = acum_metrics.get('loss', 0) + loss.item()
        acum_metrics['loss_standard'] = acum_metrics.get('loss_standard', 0) + (standard_loss.item() if standard_loss is not None else 0)
        acum_metrics['loss_attractor'] = acum_metrics.get('loss_attractor', 0) + (attractor_loss.item() if attractor_loss is not None else 0)
        acum_metrics['loss_scd'] = acum_metrics.get('loss_scd', 0) + (scd_loss.item() if scd_loss is not None else 0)
        acum_metrics['loss_seg_PIT'] = acum_metrics.get('loss_seg_PIT', 0) + (seg_PIT_loss.item() if seg_PIT_loss is not None else 0)
        
    else :
        raise ValueError(f"Invalid model type: {model_type}")
    
    return loss, acum_metrics, y_pred

# Parse command-line arguments
def parse_arguments() -> SimpleNamespace: #def dummy():
    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('-c', '--config', help='config file path',
                        action=yamlargparse.ActionConfigFile)
    parser.add_argument('--context-size', default=0, type=int)
    parser.add_argument('--dev-batchsize', default=1, type=int,
                        help='number of utterances in one development batch')
    parser.add_argument('--encoder-units', type=int,
                        help='number of units in the encoder')
    parser.add_argument('--feature-dim', type=int)
    parser.add_argument('--frame-shift', type=int)
    parser.add_argument('--frame-size', type=int)
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', default=-1, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--hidden-size', type=int,
                        help='number of units in SA blocks')
    parser.add_argument('--init-epochs', type=str, default='',
                        help='Initialize model with average of epochs \
                        separated by commas or - for intervals.')
    parser.add_argument('--init-model-path', type=str, default='',
                        help='Initialize the model from the given directory')
    parser.add_argument('--input-transform', default='',
                        choices=['logmel', 'logmel_meannorm',
                                 'logmel_meanvarnorm'],
                        help='input normalization transform')
    parser.add_argument('--log-report-batches-num', default=1, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max-epochs', type=int,
                        help='Max. number of epochs to train')
    parser.add_argument('--min-length', default=0, type=int,
                        help='Minimum number of frames for the sequences'
                             ' after downsampling.')
    parser.add_argument('--model-type', default='TransformerEDA',
                        help='Type of model (for now only TransformerEDA)')
    parser.add_argument('--noam-warmup-steps', default=100000, type=float)
    parser.add_argument('--num-frames', default=500, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-speakers', type=int,
                        help='maximum number of speakers allowed')
    parser.add_argument('--num-workers', default=1, type=int,
                        help='number of workers in train DataLoader')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--sampling-rate', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--subsampling', default=10, type=int)
    parser.add_argument('--train-batchsize', default=1, type=int,
                        help='number of utterances in one train batch')
    parser.add_argument('--train-data-dir', type=str,
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--transformer-encoder-dropout', type=float)
    parser.add_argument('--transformer-encoder-n-heads', type=int)
    parser.add_argument('--transformer-encoder-n-layers', type=int)
    parser.add_argument('--use-last-samples', default=True, type=bool)
    parser.add_argument('--vad-loss-weight', default=0.0, type=float)
    parser.add_argument('--valid-data-dir', type=str,
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('--model_type', default='TransformerEDA',type=str,)
    # Multi-GPU related addition
    parser.add_argument('--device', default="cuda", type=str,
                        help='Device to use for training (e.g., "cuda" or "cpu")')

    attractor_args = parser.add_argument_group('attractor')
    attractor_args.add_argument(
        '--time-shuffle', action='store_true',
        help='Shuffle time-axis order before input to the network')
    attractor_args.add_argument(
        '--attractor-loss-ratio', default=1.0, type=float,
        help='weighting parameter')
    attractor_args.add_argument(
        '--attractor-encoder-dropout', type=float)
    attractor_args.add_argument(
        '--attractor-decoder-dropout', type=float)
    attractor_args.add_argument(
        '--detach-attractor-loss', type=bool,
        help='If True, avoid backpropagation on attractor loss')
    
    # SSCD-related parameters
    scd_args = parser.add_argument_group('state_change_detector')
    scd_args.add_argument('--scd_loss_ratio', default=1.0, type=float, help='State Change Detector loss weight')
    scd_args.add_argument('--state_change_detector_dropout', default=0.1, type=float, help='Dropout for SSCD')

    # SSCD-based segmentation loss parameters
    seg_args = parser.add_argument_group('segmentation')
    seg_args.add_argument('--seg_PIT_loss_ratio', default=1.0, type=float, help='Segment PIT loss weight')

    args = parser.parse_args()
    return args

def get_dataloaders(args):
    """ Returns training & validation dataloaders with DistributedSampler """
    
    train_set = KaldiDiarizationDataset(
        args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,  # Apply shuffle in the dataset
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )

    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        feature_dim=args.feature_dim,
        frame_shift=args.frame_shift,
        frame_size=args.frame_size,
        input_transform=args.input_transform,
        n_speakers=args.num_speakers,
        sampling_rate=args.sampling_rate,
        shuffle=args.time_shuffle,  # Apply shuffle in the dataset
        subsampling=args.subsampling,
        use_last_samples=args.use_last_samples,
        min_length=args.min_length,
    )

    train_sampler = DistributedSampler(train_set, shuffle=True)  # Set shuffle=True here
    dev_sampler = DistributedSampler(dev_set, shuffle=False)

    batch_size_per_gpu = args.train_batchsize // dist.get_world_size()  # Adjust batch size based on the number of GPUs
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        num_workers=args.num_workers,
        worker_init_fn=_init_fn,
        collate_fn=_convert,
    )

    dev_loader = DataLoader(
        dev_set,
        batch_size=batch_size_per_gpu,
        sampler=dev_sampler,
        num_workers=1,
        worker_init_fn=_init_fn,
        collate_fn=_convert,
    )

    return train_loader, dev_loader, train_sampler, dev_sampler

# Synchronize and average across GPUs
def average_across_gpus(value):
    """ Computes the average of a value across all GPUs """
    if isinstance(value, torch.Tensor):
        tensor = value.clone().detach()
    else:
        tensor = torch.tensor(value, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))  

    if dist.is_initialized(): 
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    
    return tensor.item()

def gather_tensors(tensor, world_size):
    """ Collects data from all GPUs and returns a single list """
    tensor = tensor.to(torch.cuda.current_device())  # Move to GPU
    gathered_tensors = [torch.zeros_like(tensor, device=tensor.device) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)  # Merge into a single large tensor

if __name__ == '__main__':
    args = parse_arguments()
    
    # Multi-GPU setup
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # Assign GPU per process
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    args.device = f"cuda:{rank}"  # Dynamic GPU assignment
    
    torch.cuda.synchronize()  # GPU synchronization (only needed in multi-GPU environments)

    # Set seed for reproducibility
    setup_seed(args.seed)

    # Initialize data loaders
    train_loader, dev_loader, train_sampler, dev_sampler = get_dataloaders(args)

    # Model setup (ensuring correct GPU loading order)
    model = get_model(args)  # Create model (not yet moved to GPU)
    model.to(device)  # Move model to GPU
    torch.cuda.synchronize()  # Synchronize after moving model to GPU (for stability)

    # Ensure all model parameters are on the correct GPU before proceeding
    for name, param in model.named_parameters():
        if param.device != device:
            param.data = param.data.to(device)
        
    # Load checkpoint if available
    if args.init_model_path:
        model = average_checkpoints(device, model, args.init_model_path, args.init_epochs)
        model.to(device)  # Force move model to GPU

    # Set up optimizer
    optimizer = setup_optimizer(args, model)

    if optimizer is None:
        raise ValueError("setup_optimizer() returned None! Check optimizer initialization.")

    # Ensure all model parameters are on the correct GPU before applying DDP
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"[ERROR] Parameter {name} is on {param.device}, expected {device}")
            param.data = param.data.to(device)  # Force move to the correct device
            
    # Apply DistributedDataParallel (DDP) (only once!)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Ensure all model parameters are on the same GPU after DDP
    for name, param in model.named_parameters():
        assert param.device == device, f" Parameter {name} is still on {param.device}, expected {device}"
        
    torch.cuda.synchronize()  # Final synchronization after DDP application

    logging.info(args)
    
    if rank == 0:
        writer = SummaryWriter(f"{args.output_path}/tensorboard")
        
    train_batches_qty = len(train_loader)
    dev_batches_qty = len(dev_loader)
    
    acum_train_metrics = new_metrics()
    acum_dev_metrics = new_metrics()    

    checkpoint_dir = os.path.join(args.output_path, 'models')

    if os.path.exists(checkpoint_dir):
        checkpoints = [
            f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".tar")
        ]

        # Extract epoch numbers from checkpoint filenames
        epoch_pattern = re.compile(r"checkpoint_(\d+).tar")
        
        checkpoint_epochs = [
            (int(epoch_pattern.search(f).group(1)), f)
            for f in checkpoints if epoch_pattern.search(f)
        ]
        
        if checkpoint_epochs:
            # Select the checkpoint with the highest epoch number
            latest_epoch, latest_checkpoint = max(checkpoint_epochs, key=lambda x: x[0])
            latest = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Latest checkpoint : {latest} (Epoch {latest_epoch})")

            if os.path.exists(latest) and os.path.getsize(latest) > 0:  # Check file existence and size
                try:
                    checkpoint = torch.load(latest, map_location=args.device, weights_only=True)
                    epoch, model, optimizer, _ = load_checkpoint(args, latest)
                    init_epoch = epoch
                except Exception as e:
                    print(f" [ERROR] Failed to load checkpoint {latest}, skipping... Reason: {e}")
                    init_epoch = 0  # If checkpoint is corrupted, start from scratch
            else:
                print(f"[INFO] Checkpoint file {latest} is missing or corrupted. Training from scratch.")
                init_epoch = 0
        else:
            print("[INFO] No valid checkpoint found. Starting training from scratch.")
            init_epoch = 0
    else:
        print("[INFO] No checkpoint directory found. Starting training from scratch.")
        init_epoch = 0

    for epoch in range(init_epoch, args.max_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)  # Required when using DDP
        
        # Set epoch for DDP
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
            
         # Create progress bar only for rank 0
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}") if rank == 0 else None
        
        for i, batch in enumerate(train_loader):
            features = batch['xs']
            labels = batch['ts']

            # Auto-adjust batch size when using multi-GPU
            n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                     if t.sum() > 0 else 0 for t in labels])
            max_n_speakers = max(n_speakers)
            
            features, labels = pad_sequence(features, labels, args.num_frames)
            labels = pad_labels(labels, max_n_speakers)
            
            # Move labels to the same GPU as inputs
            features = torch.stack(features).to(device)
            labels = torch.stack(labels).to(device)

            if args.model_type == 'TransformerEDA':
                loss, acum_train_metrics, y_pred = compute_loss_and_metrics(
                    model.module if isinstance(model, torch.nn.DataParallel) else model,
                    labels, features, n_speakers, acum_train_metrics,
                    args.vad_loss_weight, args.detach_attractor_loss, args.model_type)
            elif args.model_type == 'TransformerSCDEDA':
                loss, acum_train_metrics, y_pred = compute_loss_and_metrics(
                    model.module if isinstance(model, torch.nn.DataParallel) else model,
                    labels, features, n_speakers, acum_train_metrics,
                    args.vad_loss_weight, args.detach_attractor_loss, args.model_type)
            else:
                raise ValueError(f"Invalid model type: {args.model_type}")
                        
            # Check loss value: Stop if it is 0 or NaN
            assert loss.item() > 0 and not torch.isnan(loss) and not torch.isinf(loss), \
                f" Loss value is abnormal: {loss.item()}"
            
            # Compute the average loss and metrics across all GPUs
            avg_loss = average_across_gpus(loss.item())  
            avg_metrics = {k: average_across_gpus(v) for k, v in acum_train_metrics.items()}
            
            global_step = epoch * train_batches_qty + i
            
            if global_step % 1000 == 0:  # Log every 1000 steps    
                if rank == 0:
                    for k in avg_metrics.keys():
                        writer.add_scalar(
                            f"train_{k}",
                            avg_metrics[k] / args.log_report_batches_num,
                            global_step)
                    writer.add_scalar(
                        "lrate",
                        get_rate(optimizer),
                        global_step)
                acum_train_metrics = reset_metrics(acum_train_metrics)
            optimizer.zero_grad()
            
            # Debug before calling loss.backward()
            try:
                loss.backward()
            except RuntimeError as e:
                print(f" Backward Error: {e}")
                torch.autograd.set_detect_anomaly(True)  # Enable detailed backtrace
                loss.backward()  # try again
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradclip)
            optimizer.step()

            # Update tqdm progress bar
            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=avg_loss)
        
        if rank == 0:
            progress_bar.close()  # Close tqdm progress bar
             
        save_checkpoint(args, epoch+1, model, optimizer, loss)
        
        with torch.no_grad():
            model.eval()

            acum_dev_metrics = reset_metrics(acum_dev_metrics)  # Reset validation metrics before evaluation

            for i, batch in enumerate(dev_loader):
                features = batch['xs']
                labels = batch['ts']
                n_speakers = np.asarray([max(torch.where(t.sum(0) != 0)[0]) + 1
                                        if t.sum() > 0 else 0 for t in labels])
                max_n_speakers = max(n_speakers)
                features, labels = pad_sequence(features, labels, args.num_frames)
                labels = pad_labels(labels, max_n_speakers)
                features = torch.stack(features).to(device)
                labels = torch.stack(labels).to(device)
                _, acum_dev_metrics, y_pred = compute_loss_and_metrics(
                    model, labels, features, n_speakers, acum_dev_metrics,
                    args.vad_loss_weight,
                    args.detach_attractor_loss,
                    args.model_type)
                    
        # Compute average validation metrics across all GPUs
        avg_dev_metrics = {k: average_across_gpus(v) for k, v in acum_dev_metrics.items()}

        # Log validation metrics only on rank 0
        if rank == 0:
            for k in avg_dev_metrics.keys():
                writer.add_scalar(
                    f"dev_{k}", avg_dev_metrics[k] / dev_batches_qty, epoch)  # Log per epoch