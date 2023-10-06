## How to Run

The running scripts are provided in `scripts/vpt/`. Make sure you change the path in `DATA` and run the commands under `VPT/scripts/vpt/`.

### Generalization From Base to New Classes

This corresponds to the experiments in Section 4.1, i.e., Table 1 and Figure 2.

You will need both `scripts/vpt/base2new_train.sh` and `scripts/vpt/base2new_test.sh`. The former trains a model on base classes while the latter evaluates the trained model on new classes. Both scripts have file input arguments, i.e.:
* `DATASET` (takes as input a dataset name, like `imagenet` or `caltech101`. The valid names are the files' names in `VPT/configs/datasets/`.)
* `SEED`(Seed number)
* `GPUIDS` (List of gpu ids, should be provided as a sequence of number, separated by ",")
* `L` (Number of Monte Carlo samples)
* `EPOCHS` (Number of training epochs)

To reduce the possibility of doing mistakes for reproduction, for each dataset, we provide a python script taking `GPUIDS`, `L`, and `EPOCHS` as input arguments. 

Below we provide an example on how to train and evaluate the model on all datasets.

```bash
# Caltech dataset
python train_eval_caltech.py --gpuids 0,1,2,3,4,5,6,7 --l 20 --epochs 20

# OxfordPets dataset
python train_eval_pets.py --gpuids 0,1,2,3,4,5,6,7 --l 40 --epochs 20

# StanfordCars dataset
python train_eval_cars.py --gpuids 0,1,2,3,4,5,6,7 --l 20 --epochs 40

# OxfordFlowers dataset
python train_eval_flowers.py --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 40

# Food101 dataset
python train_eval_food101.py --gpuids 0,1,2,3,4,5,6,7 --l 20 --epochs 20

# FGVCAircraft dataset
python train_eval_fgvc.py --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10

# SUN397 dataset
python train_eval_sun.py --gpuids 0,1,2,3,4,5,6,7 --l 20 --epochs 10

# DTD dataset
python train_eval_dtd.py --gpuids 0,1,2,3,4,5,6,7 --l 40 --epochs 10

# EuroSAT dataset
python train_eval_eurosat.py --gpuids 0,1,2,3,4,5,6,7 --l 20 --epochs 60

# UCF101 dataset
python train_eval_ucf101.py --gpuids 0,1,2,3,4,5,6,7 --l 5 --epochs 20

# Imagenet dataset
python train_eval_imagenet.py --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on ImageNet using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– VPT/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– imagenet/
|   |   |   |–– shots_16/
|   |   |   |   |–– VPT/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/imagenet/shots_16/VPT/vit_b16_c4_ep10_batch1_ctxv1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/imagenet/shots_16/VPT/vit_b16_c4_ep10_batch1_ctxv1 --test-log
```

### Cross-Dataset Transfer

This corresponds to the experiments in Section 4.2, i.e., Table 2.

The relevant scripts are `scripts/VPT/xd_train.sh` and `scripts/VPT/xd_test.sh` where the `DATASET` variable is set to the default, namely `imagenet`. To train the model, run

```bash
python train_cross_dataset.py --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
```

Then, you evaluate the model on other datasets, e.g.,

```bash
python train_cross_dataset.py --dname caltech101 --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname dtd --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname eurosat --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname fgvc_aircraft --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname food101 --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname oxford_flowers --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname oxford_pets --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname stanford_cars --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname sun397 --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname ucf101 --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10

```

### Domain Generalization

This corresponds to the experiments in Section 4.3, i.e., Table 3.

The steps are similar to those discussed in "Cross-Dataset Transfer" except you evaluate the model on the variants of ImageNet, i.e., `imagenetv2`, `imagenet_sketch`, `imagenet_a` and `imagenet_r`.

```bash
python train_cross_dataset.py --dname imagenetv2 --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname imagenet_sketch --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname imagenet_a --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10
python train_cross_dataset.py --dname imagenet_r --gpuids 0,1,2,3,4,5,6,7 --l 10 --epochs 10

```