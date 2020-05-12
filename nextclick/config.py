import argparse
import util


def load():
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--task', default="next_click")

    # deep neural network
    parser.add_argument('--dnn_num_epochs', default=100, type=int)
    parser.add_argument('--dnn_architecture', default=0, type=int)
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--batch_size_train', default=256, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)

    # evaluation
    parser.add_argument('--num_folds', default=2, type=int)
    parser.add_argument('--cross_validation', default=True, type=util.str2bool)

    # directories and data
    parser.add_argument('--data_set', default="data_v4_converted_no_context.csv")
    parser.add_argument('--data_dir', default="../data/")
    parser.add_argument('--checkpoint_dir', default="./checkpoints/")
    parser.add_argument('--result_dir', default="./results/")

    args = parser.parse_args()

    return args
