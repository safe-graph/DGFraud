from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
from sklearn import metrics

from supervised_models import SupervisedGraphconsis
from models import SAGEInfo
from minibatch import NodeMinibatchIterator
from neigh_samplers import UniformNeighborSampler, DistanceNeighborSampler
from utils import load_data

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '../../dataset/', 'prefix identifying training data. must be specified.')
flags.DEFINE_string('file_name', 'YelpChi.mat', 'file name for opening the .mat file')
flags.DEFINE_float('train_perc', 1., 'how many percentages of training data used for the model')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('context_dim', 0, 'Set to positive value to use context embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss], 
                        feed_dict=feed_dict_val)
    mic, mac = calc_f1(labels, node_outs_val[0])
    auc = calc_auc(labels, node_outs_val[0])
    return node_outs_val[1], mic, mac, auc, (time.time() - t_test)

# def log_dir():
#     log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
#     log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
#             model=FLAGS.model,
#             model_size=FLAGS.model_size,
#             lr=FLAGS.learning_rate)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     return log_dir

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    t_test = time.time()
    finished = False
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss], 
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores = calc_f1(labels, val_preds)
    auc_score = calc_auc(labels, val_preds)
    return np.mean(val_losses), f1_scores[0], f1_scores[1], auc_score, (time.time() - t_test)

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def train(train_data, test_data=None):

    G = train_data[0]
    features = train_data[1]
    id_map = train_data[2]
    class_map  = train_data[4]
    gs = train_data[5]
    num_relations = len(gs)
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))

    if not features is None:
        # pad with dummy zero vector  
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders(num_classes)
    minibatch_list = [NodeMinibatchIterator(g, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs) for g in gs]
    minibatch_main = NodeMinibatchIterator(G, 
            id_map,
            placeholders, 
            class_map,
            num_classes,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree, 
            context_pairs = context_pairs)
    adj_info_ph_list = [tf.placeholder(tf.int32, shape=minibatch_main.adj.shape) for i in range(num_relations)]
    adj_info_list = [tf.Variable(adj_info_ph, trainable=False, name="adj_info") for adj_info_ph in adj_info_ph_list]
    adj_info_main = adj_info_list[0]
    # print('****supervised_train****shape of adj_info**********:', adj_info.shape)

    if FLAGS.model == 'graphsage_mean':
        sampler_list = [DistanceNeighborSampler(adj_info) for adj_info in adj_info_list]
        if FLAGS.samples_3 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)] for sampler in sampler_list]
        elif FLAGS.samples_2 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)] for sampler in sampler_list]
        else:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)] for sampler in sampler_list]

        model = SupervisedGraphconsis(num_classes, placeholders, 
                                     features,
                                     adj_info_main,
                                     minibatch_main.deg,
                                     hete_layer_infos, 
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.context_dim, 
                                     num_re = num_relations,
                                     logging=True)
    elif FLAGS.model == 'gcn':
        sampler_list = [DistanceNeighborSampler(adj_info) for adj_info in adj_info_list]
        if FLAGS.samples_3 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)] for sampler in sampler_list]
        elif FLAGS.samples_2 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)] for sampler in sampler_list]
        else:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)] for sampler in sampler_list]

        model = SupervisedGraphconsis(num_classes, placeholders, 
                                     features,
                                     adj_info_main,
                                     minibatch_main.deg,
                                     layer_infos=hete_layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     concat=False,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.context_dim, num_re=num_relations,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler_list = [DistanceNeighborSampler(adj_info) for adj_info in adj_info_list]
        if FLAGS.samples_3 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)] for sampler in sampler_list]
        elif FLAGS.samples_2 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)] for sampler in sampler_list]
        else:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)] for sampler in sampler_list]

        model = SupervisedGraphconsis(num_classes, placeholders, 
                                       features,
                                       adj_info_main,
                                       minibatch_main.deg,
                                       layer_infos=hete_layer_infos, 
                                       aggregator_type="seq",
                                       model_size=FLAGS.model_size,
                                       sigmoid_loss = FLAGS.sigmoid,
                                       identity_dim = FLAGS.context_dim, num_re=num_relations,
                                       logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler_list = [DistanceNeighborSampler(adj_info) for adj_info in adj_info_list]
        if FLAGS.samples_3 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)] for sampler in sampler_list]
        elif FLAGS.samples_2 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)] for sampler in sampler_list]
        else:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)] for sampler in sampler_list]

        model = SupervisedGraphconsis(num_classes, placeholders, 
                                    features,
                                    adj_info_main,
                                    minibatch_main.deg,
                                     layer_infos=hete_layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.context_dim, num_re=num_relations,
                                     logging=True)

    elif FLAGS.model == 'graphsage_meanpool':
        sampler_list = [DistanceNeighborSampler(adj_info) for adj_info in adj_info_list]
        if FLAGS.samples_3 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)] for sampler in sampler_list]
        elif FLAGS.samples_2 != 0:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)] for sampler in sampler_list]
        else:
            hete_layer_infos = [[SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)] for sampler in sampler_list]

        model = SupervisedGraphconsis(num_classes, placeholders, 
                                    features,
                                    adj_info_main,
                                    minibatch_main.deg,
                                     layer_infos=hete_layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     sigmoid_loss = FLAGS.sigmoid,
                                     identity_dim = FLAGS.context_dim, num_re=num_relations,
                                     logging=True)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph_list[i]: minibatch_list[i].adj for i in range(num_relations)})
    
    # Train model
    
    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []

    # train_adj_info = tf.assign(adj_info, minibatch.adj)
    # val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    for epoch in range(FLAGS.epochs): 
        minibatch_main.shuffle() 

        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        while not minibatch_main.end():
            # Construct feed dictionary
            feed_dict, labels = minibatch_main.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            train_cost = outs[2]

            if iter % FLAGS.validate_iter == 0:
                # Validation
                # sess.run(val_adj_info.op)
                if FLAGS.validate_batch_size == -1:
                    ret = incremental_evaluate(sess, model, minibatch_main, FLAGS.batch_size)
                    val_cost, val_f1_mic, val_f1_mac, val_auc, duration = ret
                else:
                    ret = evaluate(sess, model, minibatch_main, FLAGS.validate_batch_size)
                    val_cost, val_f1_mic, val_f1_mac, val_auc, duration = ret
                # sess.run(train_adj_info.op)
                epoch_val_costs[-1] += val_cost

            # if total_steps % FLAGS.print_every == 0:
            #     summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print("Iter:", '%04d' % iter, 
                      "train_loss=", "{:.5f}".format(train_cost),
                      "train_f1_mic=", "{:.5f}".format(train_f1_mic), 
                      "train_f1_mac=", "{:.5f}".format(train_f1_mac), 
                      "val_loss=", "{:.5f}".format(val_cost),
                      "val_f1_mic=", "{:.5f}".format(val_f1_mic), 
                      "val_f1_mac=", "{:.5f}".format(val_f1_mac), 
                      "time=", "{:.5f}".format(avg_time))
 
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break

        if total_steps > FLAGS.max_total_steps:
                break
    
    print("Optimization Finished!")
    # sess.run(val_adj_info.op)
    ret = incremental_evaluate(sess, model, minibatch_main, FLAGS.batch_size)
    val_cost, val_f1_mic, val_f1_mac, val_auc, duration = ret
    print("Full validation stats:",
                  "loss=", "{:.5f}".format(val_cost),
                  "f1_micro=", "{:.5f}".format(val_f1_mic),
                  "f1_macro=", "{:.5f}".format(val_f1_mac),
                  "auc=", "{:.5f}".format(val_auc),
                  "time=", "{:.5f}".format(duration))
    # with open(log_dir() + "val_stats.txt", "w") as fp:
    #     fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} auc={:.5f} time={:.5f}".
    #             format(val_cost, val_f1_mic, val_f1_mac, val_auc, duration))

    # print("Writing test set stats to file (don't peak!)")
    # ret = incremental_evaluate(sess, model, minibatch_main, FLAGS.batch_size, test=True)
    # val_cost, val_f1_mic, val_f1_mac, val_auc, duration = ret
    # with open(log_dir() + "test_stats.txt", "w") as fp:
    #     fp.write("loss={:.5f} f1_micro={:.5f} f1_macro={:.5f} auc={:.5f} time={:.5f}".
    #             format(val_cost, val_f1_mic, val_f1_mac, val_auc, duration))

def main(argv=None):
    print("Loading training data..")
    # file_name = 'small_sample.mat'
    file_name = FLAGS.file_name
    train_perc = FLAGS.train_perc
    relations = ['net_rur', 'net_rtr', 'net_rsr']
    train_data = load_data(FLAGS.train_prefix, file_name, relations, train_perc)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()
