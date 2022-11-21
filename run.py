import os.path
import time
import argparse

from recbole.quick_start import run_recbole, load_data_and_model


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


def evaluate_models(config_file, dataset_name):
    model_list = ["Pop", "ItemKNN", "BPR", "NeuMF", "RecVAE", "LightGCN"]  # General
    model_list += ["FFM", "DeepFM"]  # Context-aware
    model_list += ["BERT4Rec", "GRU4Rec", "SHAN"]  # Sequential
    for model_name in model_list:
        print(f"running {model_name}...")
        start = time.time()
        result = run(model_name, f'config/{config_file}', dataset_name)
        t = time.time() - start
        print(f"It took {t / 60:.2f} mins")
        print(result)


def train_model(model_name="BERT4Rec", config_file=None, model=None, dataset=None):
    if model and dataset:
        model.fit(dataset)
        model.save_model()
    else:
        run(model_name, f'config/{config_file}', dataset)


# TODO: this function isnt fully tested yet
def predict_recommendations(model, user_id, item_id, top_k=10):
    return model.predict(user_id, item_id, top_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_evaluation', action='store_true')
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="BERT4Rec")
    parser.add_argument('--config_file', type=str, default="config_ml-100k.yaml")

    args = parser.parse_args()

    if args.run_evaluation:
        evaluate_models(args.config_file, args.dataset)
    elif args.checkpoint_path:
        config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
            model_file=f"saved/{args.checkpoint_path}")
        print(predict_recommendations(model, 1, 1))
    else:
        train_model(args.model_name, args.config_file, dataset=args.dataset)
