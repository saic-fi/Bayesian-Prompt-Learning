import os


datasets = ["caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101"]
mc       = [20,40,20,10,20,10,40,20,10,5]
epochs   = [20,10,60,10,20,40,20,40,10,20]
seeds    = [1]
shots    = [16]
GPOUIDS  = "0,1,2,3,4,5,6,7"
backbones = ['RN50', 'RN101']

for dataset, l, epoch in zip(datasets, mc, epochs):
    for backbone in backbones:
        os.system(f"bash base2new_train_arch.sh {dataset} 1 {GPOUIDS} 16 4 {l} {epoch} {backbone}")
        os.system(f"bash base2new_test_arch.sh {dataset} 1 {GPOUIDS} 16 4 {l} {epoch} {backbone}")
