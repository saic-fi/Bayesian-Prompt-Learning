import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuids", default="0,1,2,3,4,5,6,7", help="GPU ids to train model on")
    parser.add_argument(
        "--l", help="Number of monte carlo samples"
    )
    parser.add_argument(
        "--epochs", help="Number of training epochs"
    )
    args = parser.parse_args()
    
    for seed in [1, 2, 3]:
        os.system(f"bash base2new_train.sh caltech101 {seed} {args.gpuids} {args.l} {args.epochs}")
        os.system(f"bash base2new_test.sh caltech101 {seed} {args.gpuids} {args.l} {args.epochs}")
