import os.path
import time

import yaml
from recbole.quick_start import run_recbole
from classes.Dataset import Dataset


def run(model_name, config_file, dataset_name):
    if model_name in [
        "MultiVAE",
        "MultiDAE",
        "MacridVAE",
        "RecVAE",
        "GRU4Rec",
        "NARM",
        "STAMP",
        "NextItNet",
        "TransRec",
        "SASRec",
        "BERT4Rec",
        "SRGNN",
        "GCSAN",
        "GRU4RecF",
        "FOSSIL",
        "SHAN",
        "RepeatNet",
        "HRM",
        "NPE",
    ]:
        parameter_dict = {
            "neg_sampling": None,
        }
        return run_recbole(
            model=model_name,
            dataset=dataset_name,
            config_file_list=[config_file],
            config_dict=parameter_dict,
        )
    else:
        return run_recbole(
            model=model_name,
            dataset=dataset_name,
            config_file_list=[os.path.join("config", f"config_{dataset_name}.yaml")],
        )


def prepare_dataset(dataset_name):
    ds = Dataset(dataset_name)
    ds.convert_inter()
    ds.convert_user()
    ds.convert_item()
    del ds


def evaluate_models(config_file):
    # read config_file yml file and get the attribute "dataset"
    config = yaml.load(open(f"config/{config_file}", "r"), Loader=yaml.FullLoader)
    dataset_name = config["dataset"]
    prepare_dataset(dataset_name)

    # model_list = ["Pop", "ItemKNN", "BPR", "NeuMF", "RecVAE", "LightGCN"]  # General
    # model_list += ["FFM", "DeepFM"]                 # Context-aware
    model_list = ["GRU4Rec", "SHAN"]    # Sequential
    for model_name in model_list:
        print(f"running {model_name}...")
        start = time.time()
        result = run(model_name, f'config/{config_file}', dataset_name)
        t = time.time() - start
        print(f"It took {t / 60:.2f} mins")
        print(result)


if __name__ == "__main__":
    evaluate_models("config_hm.yaml")
