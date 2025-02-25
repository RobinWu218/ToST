# ToST on Vision Tasks


## Installation

Install required packages
```sh
pip install -r requirements.txt
```

## Usage

We use ```main.py``` for both training and evaluation. Make sure ```--data-path``` argument points to your ImageNet path.

For full details about all the available arguments, you can use
```sh
python main.py --help
```


### Training

For training using a single node, use the following command:

```sh
python -m torch.distributed.launch --nproc_per_node=[NUM_GPUS] --use_env main.py --model [MODEL_KEY] --batch-size [BATCH_SIZE] --drop-path [STOCHASTIC_DEPTH_RATIO] --output_dir [OUTPUT_PATH]
```
For example, the ToST-S12/16 model can be trained using the following command:

```sh
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model tost_cab_small_12_p16 --batch-size 128 --drop-path 0.05 --output_dir /experiments/tost_cab_small_12_p16/ --epochs 400
```

Alternatively use submitit for multinode training:
```sh
python run_with_submitit.py --partition [PARTITION_NAME] --nodes 2 --ngpus 8 --model tost_cab_small_12_p16 --batch-size 64 --drop-path 0.05 --job_dir /experiments/tost_cab_small_12_p16/ --epochs 400
```

To train ToST-T12/16 model, run:
```sh
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model tost_cab_tiny_12_p16 --batch-size 128 --drop-path 0.0 --output_dir /experiments/tost_cab_tiny_12_p16/ --epochs 400
```

To train ToST-M12/16 model, run:
```sh
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model tost_cab_medium_12_p16 --batch-size 128 --drop-path 0.15 --output_dir /experiments/tost_cab_medium_12_p16/ --epochs 400
```

### Evaluation
To evaluate a pretrained ToST model, use the following command: 
```sh
python main.py --eval --model <MODEL_KEY> --input-size <IMG_SIZE> [--full_crop] --pretrained <PATH>
```

For example, the ToST-S12/16 model can be evaluated using the following command:
```sh
python main.py --eval --model tost_cab_small_12_p16 --input-size 224 --full_crop --pretrained path/to/checkpoint
```

### Finetuning
To finetune a pretrained ToST model on some downstream tasks (e.g. CIFAR-10), use the following command:
```sh
python finetune.py --bs 256 --net tost_cab_small_12_p16 --opt adamW --lr 1e-4 --n_epochs 200 --randomaug 1 --data cifar10 --classes 10 --ckpt_dir /path/to/checkpoint --data_dir /path/to/cifar10

```

## Acknowledgements

This part of our repo is largely based on the great [XCiT](https://github.com/facebookresearch/xcit) project.