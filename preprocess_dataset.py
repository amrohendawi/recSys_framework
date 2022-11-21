import argparse
import importlib

from dataset_preprocessing.utils import dataset2class, click_dataset, multiple_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--interaction_type', type=str, default=None)
    parser.add_argument('--duplicate_removal', action='store_true')

    parser.add_argument('--convert_inter', action='store_true')
    parser.add_argument('--convert_item', action='store_true')
    parser.add_argument('--convert_user', action='store_true')

    args = parser.parse_args()

    input_args = [f"datasets/{args.dataset}", f"training_data/{args.dataset}"]
    dataset_class_name = dataset2class[args.dataset.lower()]
    dataset_class = getattr(importlib.import_module('dataset_preprocessing.Dataset'), dataset_class_name)
    if dataset_class_name in multiple_dataset:
        input_args.append(args.interaction_type)
    if dataset_class_name in click_dataset:
        input_args.append(args.duplicate_removal)
    datasets = dataset_class(*input_args)

    if args.convert_inter:
        datasets.convert_inter()
    if args.convert_item:
        datasets.convert_item()
    if args.convert_user:
        datasets.convert_user()
