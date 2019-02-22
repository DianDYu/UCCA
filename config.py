import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debugging",
                        action='store_true',
                        help="debugging mode. if not, train on training set and evaluate on dev set")
    parser.add_argument("--testing",
                        action='store_true',
                        help="testing mode for final result on the test set")
    parser.add_argument("--reading_data",
                        action='store_true',
                        help="if true, read data from passages; other wise load preprocessed data in torch file")
    parser.add_argument("--use_lowercase",
                        action='store_true',
                        help="lowercase all the tokens")
    parser.add_argument("--replace_digits",
                        action='store_true',
                        help="replace all the digits with 0")
    parser.add_argument("--unroll",
                        action='store_true',
                        help="use previous hidden state as the init hidden state in the sub_lstm_model")
    parser.add_argument("--use_embedding",
                        action='store_true',
                        help="whether to use pretrained emebdding")
    parser.add_argument("--use_both_ends",
                        action='store_true',
                        help="whether to use both ends (concat) in the sub_lstm_model")
    parser.add_argument("--not_save",
                        action='store_true',
                        help="not save trained models. used for debugging")
    parser.add_argument('--do_val',
                        action='store_true',
                        help="whether to split the training set for validation during training. Used for debugging. "
                             "Otherwise use the whole training set for both training and validation")
    parser.add_argument('--shuffle_val',
                        action='store_true',
                        help="whether to shuffle data before splitting train and val"
                             "if false, just the first num sents as training and the rest for val")
    parser.add_argument("--predict_remote",
                        action='store_true',
                        help="whether to predict the remote edges")
    parser.add_argument("--ignore_error",
                        action='store_true',
                        help="whether to ingore errors during training")
    parser.add_argument("--save_dir",
                        type=str,
                        default="models/debugging",
                        help="directory to save trained models")
    parser.add_argument("--load_dir",
                        type=str,
                        default="models/debugging",
                        help="directory to load trained models")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help="number of epochs to train")
    parser.add_argument('--is_server',
                        action='store_true',
                        help="for tuning only. chnage the diretory if running on server")
    parser.add_argument('--testing_dev',
                        action='store_true',
                        help="train on all the training set and test on the dev set")
    return parser.parse_args()
