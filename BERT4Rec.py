import os.path
from logging import getLogger

import torch
import pickle

from recbole.config import Config
from recbole.utils import init_logger, get_model, get_trainer, init_seed, set_color
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders


# from recbole.model.sequential_recommender.bert4rec import BERT4Rec


class BERT4Rec:
    def __init__(self, dataset):
        self.dataset = dataset
        self.model = None

    def run(self, saved=True, config_file_list=None, config_dict=None):
        # configurations initialization
        config = Config(model="BERT4Rec", dataset=self.dataset, config_file_list=config_file_list,
                        config_dict=config_dict)
        init_seed(config['seed'], config['reproducibility'])

        # logger initialization
        init_logger(config)
        logger = getLogger()
        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        init_seed(config['seed'], config['reproducibility'])
        self.model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
        logger.info(self.model)

        # model = model.to(device)

        # load trainer
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, self.model)

        # train model
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved,
                                                          show_progress=config['show_progress'])

        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

        logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
        logger.info(set_color('test result', 'yellow') + f': {test_result}')

        return {
            'best_valid_score': best_valid_score,
            'valid_score_bigger': config['valid_metric_bigger'],
            'best_valid_result': best_valid_result,
            'test_result': test_result
        }

    def predict(self, user, item):
        return self.model.predict(user, item)

    def save(self, path):
        torch.save(self.model, path)


if __name__ == '__main__':
    dataset_name = "hm"
    bert = BERT4Rec(dataset_name)
    bert.run(config_file_list=["config/config_hm.yaml"], config_dict={"neg_sampling": None})
    bert.save("bert4rec.pkl")
