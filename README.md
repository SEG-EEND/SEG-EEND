 # SEG-EEND Project

This repository builds upon the PyTorch implementation of End-to-End Neural Diarization (EEND) originally developed by BUT, and extends it to implement the SEG-EEND model.

SEG-EEND introduces a segment-level approach for improving diarization performance, particularly in multi-speaker environments.

---

## Usage

### Training

To run training, use the following command:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDs> <path_to_conda_python> -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=<NUM_GPUs> \
    train.py -c <path_to_config.yaml>
```

**Example:**

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 /home/SEG-EEND/anaconda3/envs/seg_eend/bin/python \
    -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    train.py -c /home/SEG-EEND/SEG-EEND/examples/train_SEG-EEND.yaml
```

> **Note:**  
> Be sure to set the **training** and **validation** data directories, as well as the **output** directory, in the configuration file.

---

### Fine-tuning

To fine-tune a pre-trained model, the process is similar.  
Make sure to set the path to the checkpoint either in the config file or via command-line arguments.

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDs> <path_to_conda_python> -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=<NUM_GPUs> \
    train.py -c <path_to_fine-tuning_config.yaml>
```

---

### Inference

To run inference, use the following command:

```bash
python eend/infer.py -c examples/infer.yaml --gpu -1
```

> **Note:**  
> Be sure to define the **data directory**, **model checkpoint**, and **output directory** in the configuration file.

---

## References

[1] S. Horiguchi, Y. Fujita, S. Watanabe, Y. Xue, and P. Garcia,  
“Encoder-decoder based attractors for end-to-end neural diarization,”  
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, vol. 30, pp. 1493–1507, 2022.

[2] Official EEND repository by BUT:  
https://github.com/BUTSpeechFIT/EEND