import argparse
import time

import numpy as np
import tensorflow as tf

import classifiers.inception
import constants
import data
import logger

# Log information
print('TF ', tf.__version__)
print('GPU', tf.config.list_physical_devices('GPU'))

# Unique ID
exp_id = str(time.time()).replace('.', '') # int(time.time()*10)

# Parse arguments
model_mapping = {
    'inception': classifiers.inception.get_model,
    # 'lite':  classifiers.lite.get_model,
}
data_mapping = {
    'tsc_ucr': constants.UNIVARIATE_DATASET_NAMES_2018,
    'mtsc_uea': constants.MULTIVARIATE_DATASET_NAMES_2018,
    'monster': constants.MONSTER_DATASET_NAMES_2025,
    **{k: [k] for k in constants.MONSTER_DATASET_NAMES_2025},
}
precision_mapping = {
    'None': None,
    'mixed_float16': 'mixed_float16',
    'mixed_bfloat16': 'mixed_bfloat16',
    'float16': 'float16',
    'bfloat16': 'bfloat16',
}
parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--model',  type=str, required=True,
                    choices=model_mapping.keys())
parser.add_argument('--dataset', type=str, required=True, nargs='+',
                    choices=data_mapping.keys())
parser.add_argument('--results', type=str, default='results',
                    help='Folder to save results')
parser.add_argument('--precision', type=str, default='None', choices=precision_mapping.keys())
args = parser.parse_args()

model_name = args.model
model_builder = model_mapping.get(model_name)
datasets = []
for d_name in args.dataset:
    datasets += data_mapping.get(d_name)
results = args.results
precision = precision_mapping.get(args.precision)

if precision is not None:
    policy = tf.keras.mixed_precision.Policy(precision)
    print(f'Compute dtype: {policy.compute_dtype}')
    print(f'Variable dtype: {policy.variable_dtype}')
    tf.keras.mixed_precision.set_global_policy(policy)

# Columns to log
columns = {'model': str, 'dataset': str, 'train_time_s': int,
           'accuracy': float, 'f1_score': float, 'loss': float,
           'train_accuracy': float, 'train_f1_score': float, 'train_loss': float}
# Logger
logger = logger.CSV(f'{results}/EXP_{args.model}_{exp_id}.csv', columns)


def train(x_train, y_train, n_classes, dataset=''):
    codename = f'{dataset}_'
    input_shape = (None, x_train.shape[-1])

    model = model_builder(n_classes=n_classes, input_shape=input_shape)

    '''
    for layer in model.layers:
        print(layer.name, layer.dtype, layer.compute_dtype)
    '''

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(), 
                  metrics=[tf.keras.metrics.CategoricalAccuracy(),
                           tf.keras.metrics.F1Score(average='micro')])
    
    model.save(f'models/{codename}{exp_id}_best.keras')
    
    # callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'models/{codename}{exp_id}_best.keras', monitor='loss', save_best_only=True)
    callbacks = [reduce_lr, model_checkpoint]

    # train parameters
    batch_size = 16
    nb_epochs = 2000

    if dataset in constants.MONSTER_DATASET_NAMES_2025_FAST:
        nb_epochs = 100
        print(f'reduce nb_epochs to {nb_epochs} ({dataset})')
    if dataset in constants.MONSTER_DATASET_NAMES_2025_SLOW:
        nb_epochs = 10
        print(f'reduce nb_epochs to {nb_epochs} ({dataset})')

    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))

    # run train
    hist = model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs, callbacks=callbacks, verbose=0)
    
    model.save(f'models/{codename}{exp_id}_last.keras')

    return model


for dataset_name in datasets:
    x_train, y_train, x_test, y_test = data.load_dataset(dataset_name)
    length_ts = int(x_train.shape[1])
    n_classes = int(y_train.shape[1])

    t0 = time.time()
    model = train(x_train, y_train, n_classes, dataset=dataset_name)
    t1 = time.time()
    
    train_metrics = model.evaluate(x_train, y_train, return_dict=True, verbose=0)
    metrics = model.evaluate(x_test, y_test, return_dict=True, verbose=0)
    
    print(f"{metrics['categorical_accuracy']:.4f} {metrics['f1_score']:.4f} {metrics['loss']:.4f} {dataset_name}")

    logger.add({'model': model_name, 'dataset': dataset_name, 'train_time_s': int(t1 - t0),
           'accuracy': metrics['categorical_accuracy'], 'f1_score': metrics['f1_score'], 'loss': metrics['loss'],
           'train_accuracy': train_metrics['categorical_accuracy'], 'train_f1_score': train_metrics['f1_score'], 'train_loss': train_metrics['loss']})
    