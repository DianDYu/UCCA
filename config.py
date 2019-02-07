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
    # parser.add_argument("--testing",
    #                     action='store_true',
    #                     help="testing mode for final result on the test set")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")

    return parser.parse_args()
