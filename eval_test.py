import argparse
import time

import numpy as np
import tensorflow as tf

import classifiers.inception
import constants
import data
import logger
import quantizer

# Log information
print('TF ', tf.__version__)
print('GPU', tf.config.list_physical_devices('GPU'))

# Parse arguments
model_mapping = {
    'inception': classifiers.inception.get_model,
    # 'lite':  classifiers.lite.get_model,
}
data_mapping = {
    'tsc_ucr': constants.UNIVARIATE_DATASET_NAMES_2018,
    'mtsc_uea': constants.MULTIVARIATE_DATASET_NAMES_2018,
    'monster': constants.MONSTER_DATASET_NAMES_2025,
}
parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--model',  type=str, required=True,
                    choices=model_mapping.keys())
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    choices=data_mapping.keys())
parser.add_argument('--results', type=str, default='results',
                    help='Folder to save results')
parser.add_argument('--expid', type=str, default=str(time.time()).replace('.', ''),
                    help='Experience unique ID')
args = parser.parse_args()

model_name = args.model
model_builder = model_mapping.get(model_name)
datasets = []
for d_name in args.dataset:
    datasets += data_mapping.get(d_name)
results = args.results
exp_id = args.expid
# Columns to log
columns = {'model': str, 'dataset': str,
           'accuracy': float, 'f1_score': float, 'loss': float,
           'train_accuracy': float, 'train_f1_score': float, 'train_loss': float,
           'eval_time_total': float}
# Logger
logger = logger.CSV(f'{results}/BASELINE_{args.model}_{exp_id}.csv', columns) # _IDENTITY

for dataset_name in datasets:
    x_train, y_train, x_test, y_test = data.load_dataset(dataset_name)
    length_ts = int(x_train.shape[1])
    n_classes = int(y_train.shape[1])

    model = tf.keras.models.load_model(f'models/{dataset_name}_{exp_id}_best.keras')
    
    # model = quantizer.apply_to_conv(model, qtz=quantizer.identity_quantize)

    t0 = time.perf_counter()

    train_metrics = model.evaluate(x_train, y_train, return_dict=True, verbose=0)
    metrics = model.evaluate(x_test, y_test, return_dict=True, verbose=0)

    t1 = time.perf_counter()
    eval_time_total = t1 - t0
    
    print(f"{metrics['categorical_accuracy']:.4f} {metrics['f1_score']:.4f} {metrics['loss']:.4f} {dataset_name}")

    logger.add({'model': model_name, 'dataset': dataset_name,
           'accuracy': metrics['categorical_accuracy'], 'f1_score': metrics['f1_score'], 'loss': metrics['loss'],
           'train_accuracy': train_metrics['categorical_accuracy'], 'train_f1_score': train_metrics['f1_score'], 'train_loss': train_metrics['loss'],
           'eval_time_total': eval_time_total})
