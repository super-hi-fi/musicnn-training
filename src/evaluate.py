import argparse
import json
import pescador
import shared, train
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
from argparse import Namespace

config_file = Namespace(**yaml.load(open('config_file.yaml'), Loader=yaml.SafeLoader))

def evaluation(batch_dispatcher, tf_vars, array_cost, pred_array, id_array):

    [sess, normalized_y, cost, x, y_, is_train] = tf_vars
    for batch in tqdm(batch_dispatcher):
        pred, cost_pred = sess.run([normalized_y, cost], feed_dict={x: batch['X'], y_: batch['Y'], is_train: False})
        if not array_cost: # if array_cost is empty, is the first iteration
            pred_array = pred
            id_array = batch['ID']
        else:
            pred_array = np.concatenate((pred_array,pred), axis=0)
            id_array = np.append(id_array,batch['ID'])
        array_cost.append(cost_pred)
    print('predictions', pred_array.shape)
    print('cost', np.mean(array_cost))
    return array_cost, pred_array, id_array


if __name__ == '__main__':

    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', nargs='+', help='List of models to evaluate', required=True)
    parser.add_argument('-c', '--config', help='The (optional) path for a configuration YAML')
    args = parser.parse_args()

    # pull in an optional configuration file (to avoid changing the one in the repo)
    if (args.config):
        config_file = Namespace(**yaml.load(open(args.config), Loader=yaml.FullLoader))

    TEST_BATCH_SIZE = 64
    FILE_INDEX = config_file.DATA_FOLDER + config_file.config_train['spec']['audio_representation_folder'] + 'index.tsv'
    FILE_GROUND_TRUTH_TEST = config_file.config_train['spec']['gt_test']
    FOLD = config_file.config_train['spec']['fold']

    models = args.list

    # load all audio representation paths
    [audio_repr_paths, id2audio_repr_path] = shared.load_id2path(FILE_INDEX)

    # load ground truth
    [ids, id2gt] = shared.load_id2gt(FILE_GROUND_TRUTH_TEST)
    print('groundtruth file: {}'.format(FILE_GROUND_TRUTH_TEST))
    print('# Test set size: ', len(ids))

    array_cost, pred_array, id_array = [], None, None

    for model in models:

        experiment_folder = config_file.MODEL_FOLDER + 'experiments/' + str(model) + '/'
        config = json.load(open(experiment_folder + 'config.json'))
        print('Experiment: ' + str(model))
        print('\n' + str(config))

        # pescador: define (finite, batched & parallel) streamer
        pack = [config, 'overlap_sampling', config['n_frames'], False]
        streams = [pescador.Streamer(train.data_gen_abs_path, id, config_file.DATA_FOLDER + id2audio_repr_path[id], id2gt[id], pack) for id in ids]
        mux_stream = pescador.ChainMux(streams, mode='exhaustive')
        batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE, partial=True)
        batch_streamer = pescador.ZMQStreamer(batch_streamer)

        # tensorflow: define model and cost
        fuckin_graph = tf.Graph()
        with fuckin_graph.as_default():
            sess = tf.Session()
            [x, y_, is_train, y, normalized_y, cost, model_vars] = train.tf_define_model_and_cost(config)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            results_folder = experiment_folder
            saver.restore(sess, results_folder)
            tf_vars = [sess, normalized_y, cost, x, y_, is_train]
            array_cost, pred_array, id_array = evaluation(batch_streamer, tf_vars, array_cost, pred_array, id_array)
            sess.close()

    print('Predictions computed, now evaluating..')
    y_true, y_pred = shared.average_predictions(pred_array, id_array, ids, id2gt)

    roc_auc, pr_auc = shared.auc_with_aggergated_predictions(y_true, y_pred)

    acc = shared.compute_accuracy(y_true, y_pred)

    # print experimental results
    print('\nExperiment: ' + str(models))
    print(config)
    print('ROC-AUC: ' + str(roc_auc))
    print('PR-AUC: ' + str(pr_auc))
    print('Acc: ' + str(acc))
    # store experimental results
    to = open(config_file.MODEL_FOLDER + 'results_{}'.format(FOLD), 'w')
    to.write('Experiment: ' + str(models))
    to.write('\nROC AUC: ' + str(roc_auc))
    to.write('\nPR AUC: ' + str(pr_auc))
    to.write('\nAcc: ' + str(acc))
    to.write('\n')
    to.close()

    predictions = {id: list(pred.astype('float64')) for id, pred in zip(ids, y_pred)}
    predictions_file = config_file.MODEL_FOLDER + 'predictions_{}.json'.format(FOLD)

    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
