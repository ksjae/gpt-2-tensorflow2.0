import glob
import json
import click
from data_pipeline import input_fn
from gpt2_model import *
from tensorflow.keras.mixed_precision import experimental as mixed_precision

_ROOT = "gs://kogpt2"
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=2048, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=50257, show_default=True, help="Vocab size")
@click.option('--optimizer', type=str, default="adafactor", show_default=True, help="optimizer type")
@click.option('--batch-size', type=int, default=1, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
@click.option('--graph-mode', type=bool, default=False, show_default=True, help="TF run mode")
@click.option('--distributed', type=bool, default=False, show_default=True, help="distributed training")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          optimizer, batch_size, learning_rate, graph_mode, distributed):
    par_map = {"num_layers": num_layers, "d_model": embedding_size,
               "num_heads": num_heads, "dff": dff,
               "max_seq_len": max_seq_len, "vocab_size": vocab_size}

    global_batch_size=8
    tpu = True
    # exp_name = "_".join(['{}_{}'.format(k, v) for k, v in par_map.items()])

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with tf.io.gfile.GFile(MODEL_DIR + '/model_par.json', 'w') as f:
        json.dump(par_map, f)

    tf_records = tf.io.gfile.glob(_ROOT + "/data/tf_records/*")
    if distributed:
        dist_dataset = input_fn(tf_records, batch_size=batch_size * global_batch_size)
        if tpu:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='v3')
            tf.config.experimental_connect_to_cluster(resolver)
            tf.keras.backend.clear_session()
            tf.tpu.experimental.shutdown_tpu_system(resolver) # Clear cache
            context = tf.tpu.experimental.initialize_tpu_system(resolver)
            mirrored_strategy = tf.distribute.TPUStrategy(resolver)
            policy = mixed_precision.Policy('mixed_bfloat16')
            mixed_precision.set_policy(policy)
        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(dist_dataset)
        with mirrored_strategy.scope():
            model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size, global_batch_size,
                         optimizer=optimizer, learning_rate=learning_rate)
            model.create_optimizer()
            model.create_checkpoint_manager(MODEL_DIR)
            model.create_summary_writer(LOG_DIR)

        model.mirrored_strategy = mirrored_strategy
        model.compile()
        model.fit(dist_dataset, graph_mode)
    else:
        dataset = input_fn(tf_records, batch_size=batch_size)
        model = Gpt2(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
                     optimizer=optimizer, learning_rate=learning_rate)
        model.create_optimizer()
        model.create_checkpoint_manager(MODEL_DIR)
        model.create_summary_writer(LOG_DIR)
        model.fit(dataset, graph_mode)
        print("Training Done................")


if __name__ == "__main__":
    train()
