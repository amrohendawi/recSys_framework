# Recommender Systems Implementation

This repository provides a general framework for preparing datasets and building various recommender systems. It was
mainly used for evaluating different RecSys models and also for training BERT4Rec model based on the original paper [1].

The model is implemented in PyTorch.

## Getting Started

This repository is tested on Python 3.9 and PyTorch 1.12.0 with Cuda 11.6.

To prepare the working environment, please run the following command:

1. Clone the repository

```bash
git clone https://github.com/amrohendawi/recSys_framework

cd recommender_systems
```

2. Create a virtual environment and activate it

```bash
python3 -m venv venv

source venv/bin/activate
```

3. Install the requirements

```bash
pip install -r requirements.txt
```

## Description of the Code Structure

The source code is divided into the following files/folders:

- [`config`](config): contain configuration files specific to a model/dataset/evaluation.
- [`saved`](saved): contain saved models and dataloaders.
- [`datasets`](datasets): contain raw dataset files that still need to be processed.
- [`training_data`](training_data): contain processed atomic formatted dataset files that can be fed into the model.
- [`log`](log) contains the log file of the training process.
- [`log_tensorboard`](log_tensorboard) contains the tensorboard log file of the training process.
- [`wandb`](wandb) contains the wandb log file of the training process.
- [`preprocess_dataset.py`](preprocess_dataset.py): preprocess the raw dataset files into atomic formatted dataset
  files.
- [`run.py`](run.py): run/train/evaluate models.

## Dataset

The datasets used in the examples are the [H&M](https://www.kaggle.com/datasets/shionhonda/hm-recbole-atomic-files)
and [ML-100k](https://grouplens.org/datasets/movielens/100k/) datasets. The raw dataset files must be stored in the
[`datasets`](datasets) folder. The atomic formatted dataset files are stored after running `preprocess_dataset.py` in
the [`training_data`](training_data).

To use a different dataset, it must be in atomic format.

<details>
<summary>A list of datasets discussed in this repo</summary>

## General Datasets

| SN | Dataset        | Instructions |
|----|----------------|--------------|
| 1  | MovieLens      |[Link](usage/MovieLens.md)|
| 2  | Anime          |[Link](usage/Anime.md)|
| 3  | Epinions       |[Link](usage/Epinions.md)|
| 4  | Yelp           |[Link](usage/Yelp.md)|
| 5  | Netflix        |[Link](usage/Netflix.md)|
| 6  | Book\-Crossing |[Link](usage/Book-Crossing.md)|
| 7  | Jester         |[Link](usage/Jester.md)|
| 8  | Douban         |[Link](usage/Douban.md)|
| 9  | Yahoo Music    |[Link](usage/YahooMusic.md)|
| 10 | KDD2010        |[Link](usage/KDD2010.md)|
| 11 | Amazon         |[Link](usage/Amazon.md)|
| 12 | Pinterest      |[Link](usage/Pinterest.md)|
| 13 | Gowalla        |[Link](usage/Gowalla.md)|
| 14 | Last\.FM       |[Link](usage/LastFM.md)|
| 15 | DIGINETICA     |[Link](usage/DIGINETICA.md)|
| 16 | Steam          |[Link](usage/Steam.md)|
| 17 | Ta Feng        |[Link](usage/TaFeng.md)|
| 18 | Foursquare     |[Link](usage/Foursquare.md)|
| 19 | Tmall          |[Link](usage/Tmall.md)|
| 20 | YOOCHOOSE      |[Link](usage/YOOCHOOSE.md)|
| 21 | Retailrocket   |[Link](usage/Retailrocket.md)|
| 22 | LFM\-1b        |[Link](usage/LFM-1b.md)|
| 23 | MIND           |[Link](usage/MIND.md)|

### CTR Datasets

| SN | Dataset           | Instructions |
|----|-------------------|:------------:|
| 1  | Criteo            |[Link](usage/Criteo.md)|
| 2  | Avazu             |[Link](usage/Avazu.md)|
| 3  | iPinYou           |[Link](usage/iPinYou.md)|
| 4  | Phishing websites |[Link](usage/Phishing%20Websites.md)|
| 5  | Adult             |[Link](usage/Adult.md)|

### Knowledge-aware Datasets

| SN | Dataset            | Instructions |
|----|--------------------|--------------|
| 1  | MovieLens          |[Link](usage/MovieLens-KG.md)|
| 2  | Amazon\-book       |[Link](usage/Amazon-book-KG.md)|
| 3  | LFM\-1b \(tracks\) |[Link](usage/LFM-1b-KG.md)|

</details>

Here is a list of RecBole datasets that can be formatted to atomic format sorted by importance:

1. [Official Recbole Datasets and Preprocessing](https://github.com/RUCAIBox/RecSysDatasets)
2. [Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj)
3. [Official list from Recbole](https://recbole.io/dataset_list.html)

If you want to use a dataset that is not in atomic format, refer to
the [RecBole documentation](https://recbole.io/docs/user_guide/data_preparation.html) or [6] to convert it to atomic
format.

## Configuration

The configuration file is located in `config` folder. The configuration defines the model hyperparameters, the dataset,
the training/evaluation parameters and more.

## Training

Here's an example on how to train a model
with [ml-100k](https://github.com/RUCAIBox/RecSysDatasets/blob/master/conversion_tools/usage/MovieLens.md) dataset:

1. Download a dataset from the dataset section and put it in the `datasets` folder:

```bash
cd datasets
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

2. Preprocess the dataset to generate the atomic formatted dataset files:

```bash
python preprocess_dataset.py --dataset ml-100k  --convert_inter --convert_item --convert_user
```

These files will then be stored in the `training_data` folder automatically.

4. Prepare a configuration file for the model similar to the one in `config/config_ml-100k.yaml`.
   You mainly need to tell the model where to find the dataset files and how to process them (which columns to pick from
   the .inter, .item, and .user files).


5. For training a new model on ml-100k dataset run the following script:

```bash
python run.py --model_name BERT4Rec --config config_ml-100k.yaml --dataset ml-100k
```

## Logging

There are two ways to log the training process:

1. Using [Weights & Biases](https://wandb.ai/site) (recommended). To use it, refer it to the
   [official recbole documentation](https://recbole.io/docs/user_guide/usage/use_weights_and_biases.html?highlight=wand)

2. Using [Tensorboard](https://www.tensorflow.org/tensorboard). Recbole produces tensorboard logs by default out of the
   box. To review the logs, you need to install the `tensorboard` package. Then, run the following command:

```bash
tensorboard --logdir=tensorboard_logs
```

## Evaluation

The following 2 tables show an extensive evaluation of different recommender models
on [H&M](https://www.kaggle.com/datasets/shionhonda/hm-recbole-atomic-files) dataset.

<details>
<summary>1. Evaluation results</summary>

| index | model              | type          | duration | best_valid_score | valid_score | recall | MRR    | nDCG   | hit | precision | map |
|-------|--------------------| ------------- |----------| ---------------- | ------------------ |--------|--------|--------| ------ |-----------| ------ |
| 0     | Pop                | general       | 1.48     | 0.4703           | True        | 0.6663 | 0.572  | 0.5227 | 0.8842 | 0.1457    | 0.408  |
| 1     | ItemKNN            | general       | 5.45     | 0.2129           | True        | 0.5762 | 0.291  | 0.3158 | 0.8231 | 0.1041    | 0.1909 |
| 2     | BPR                | general       | 3.4      | 0.2646           | True        | 0.412  | 0.3542 | 0.3075 | 0.6366 | 0.0877    | 0.2238 |
| 3     | NeuMF              | general       | 4.07     | 0.4333           | True        | 0.6573 | 0.5276 | 0.4928 | 0.8849 | 0.1402    | 0.3733 |
| 4     | RecVAE             | general       | 85.48    | 0.4678           | True        | 0.6706 | 0.5688 | 0.5209 | 0.8922 | 0.1453    | 0.4039 |
| 5     | LightGCN           | general       | 118.88   | 0.3259           | True        | 0.4859 | 0.4039 | 0.3694 | 0.6709 | 0.1041    | 0.2809 |
| 6     | FFM                | context-aware | 5.25     | 0.1766           | True        | 0.5615 | 0.2507 | 0.2908 | 0.8036 | 0.0988    | 0.1673 |
| 7     | DeepFM             | context-aware | 5.2      | 0.1772           | True        | 0.5625 | 0.2496 | 0.2907 | 0.8046 | 0.0991    | 0.1669 |
| 8     | BERT4Rec(2 layers) | sequential    | 22.06    | 0.4363           | True        | 0.6969 | 0.5409 | 0.5157 | 0.9018 | 0.1427    | 0.3929 |
| 9     | BERT4Rec(4 layers) | sequential    | 29.42    | 0.4631           | True        | 0.7461 | 0.7952 | 0.5884 | 0.9515 | 0.5502    | 0.4631 |
| 10    | GRU4Rec            | sequential    | 5.86     | 0.5854           | True        | 0.7086 | 0.6778 | 0.6037 | 0.9038 | 0.1591    | 0.4989 |
| 11    | SHAN               | sequential    | 10.22    | 0.5201           | True        | 0.5624 | 0.4984 | 0.4706 | 0.6555 | 0.1076    | 0.4025 |

</details>

<details>
<summary>2. Test results</summary>

| index | model              | type          | duration(m) | best_valid_score | valid_score | Recall | MRR    | nDCG   | hit | precision | map    |
|-------|--------------------| ------------- |-------------| ---------------- | ------------------ |--------|--------|--------| ------ |-----------|--------|
| 0     | Pop                | general       | 1.48        | 0.4703           | True        | 0.7485 | 0.6272 | 0.5904 | 0.9346 | 0.1654    | 0.4703 |
| 1     | ItemKNN            | general       | 5.45        | 0.2129           | True        | 0.6178 | 0.3241 | 0.3461 | 0.8665 | 0.1145    | 0.2129 |
| 2     | BPR                | general       | 3.4         | 0.2646           | True        | 0.4645 | 0.398  | 0.3533 | 0.6814 | 0.1004    | 0.2646 |
| 3     | NeuMF              | general       | 4.07        | 0.4333           | True        | 0.7441 | 0.5835 | 0.5606 | 0.9403 | 0.1606    | 0.4333 |
| 4     | RecVAE             | general       | 85.48       | 0.4678           | True        | 0.7562 | 0.626  | 0.5905 | 0.9421 | 0.1654    | 0.4678 |
| 5     | LightGCN           | general       | 118.88      | 0.3259           | True        | 0.5443 | 0.4422 | 0.4176 | 0.7012 | 0.1175    | 0.3259 |
| 6     | FFM                | context-aware | 5.25        | 0.1766           | True        | 0.5986 | 0.2638 | 0.3089 | 0.8448 | 0.1074    | 0.1766 |
| 7     | DeepFM             | context-aware | 5.2         | 0.1772           | True        | 0.6016 | 0.2637 | 0.3102 | 0.8478 | 0.1083    | 0.1772 |
| 8     | BERT4Rec(2 layers) | sequential    | 22.06       | 0.4363           | True        | 0.7441 | 0.5898 | 0.5622 | 0.9284 | 0.1582    | 0.4363 |
| 9     | BERT4Rec(4 layers) | sequential    | 29.42       | 0.4631           | True        | 0.7839 | 0.7952 | 0.5884 | 0.8959 | 0.5502    | 0.3871 |
| 10    | GRU4Rec            | sequential    | 5.86        | 0.5854           | True        | 0.8045 | 0.7160 | 0.5129 | 0.8898 | 0.4795    | 0.3852 |
| 11    | SHAN               | sequential    | 10.22       | 0.5201           | True        | 0.7063 | 0.6325 | 0.6012 | 0.7979 | 0.1475    | 0.5201 |

</details>

## References

[1]: [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1904.06690)

[2] [RecBole: A Toolkit for Large-scale Recommendation System](https://arxiv.org/abs/2009.06732)

[3] [RecBole GitHub](https://github.com/RUCAIBox/RecBole)

[4] [RecBole Datasets](https://github.com/RUCAIBox/RecSysDatasets)

[5] [RecBole Tutorial](https://recbole.io/docs/index.html)

[6] [RecBole Dataset Conversion Tools](https://github.com/RUCAIBox/RecSysDatasets/tree/master/conversion_tools)
