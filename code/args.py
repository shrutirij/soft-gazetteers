"""
Author: Shruti Rijhwani
Contact: srijhwan@andrew.cmu.edu

Please cite: 
Soft Gazetteers for Low-Resource Named Entity Recognition (ACL 2020)
https://www.aclweb.org/anthology/2020.acl-main.722
"""
import argparse


def return_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed", default=128, type=int, help="word embedding size")
    parser.add_argument(
        "--word_hidden_size", default=256, type=int, help="word lstm hidden state size"
    )
    parser.add_argument(
        "--trainer", default="sgd", help="training algorithm: sgd or adam"
    )
    parser.add_argument(
        "--batch_size", default=16, type=int, help="minibatch size for training"
    )
    parser.add_argument("--lr", default=0.015, help="learning rate")
    parser.add_argument(
        "--no_decay",
        action="store_true",
        default=False,
        help="boolean hyperparameter to turn on/off learning rate decay",
    )

    parser.add_argument(
        "--autoencoder",
        action="store_true",
        help="boolean hyperparameter to turn on/off the autoencoder reconstruction loss",
    )
    parser.add_argument(
        "--crf_feats",
        action="store_true",
        help="boolean hyperparameter to turn on/off the soft gazetteer features at the crf level",
    )
    parser.add_argument(
        "--lstm_feats",
        action="store_true",
        help="boolean hyperparameter to turn on/off the soft gazetteer features at the lstm level",
    )
    parser.add_argument(
        "--feat_func",
        default="tanh",
        help="non-linearity for feature vector transformation: relu or tanh",
    )

    parser.add_argument("--train", help="train file in conll format")
    parser.add_argument("--dev", help="validation file in conll format")
    parser.add_argument("--test", help="test file in conll format")
    parser.add_argument(
        "--train_feats", default=None, help="npz file with train file features"
    )
    parser.add_argument(
        "--dev_feats", default=None, help="npz file with dev file features"
    )
    parser.add_argument(
        "--test_feats", default=None, help="npz file with test file features"
    )

    parser.add_argument(
        "--testing", default=False, action="store_true", help="testing only"
    )
    parser.add_argument(
        "--restart",
        default=False,
        action="store_true",
        help="restart training (fine-tuning an existing model)",
    )

    parser.add_argument("--output_name", help="name for log file and saved model")

    return parser
