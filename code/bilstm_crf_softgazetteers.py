"""
CNN-BiLSTM-CRF model (Ma and Hovy, 2016) with soft gazetteer features and an autoencoder reconstruction loss (Wu et al., 2018)

Author: Shruti Rijhwani
Contact: srijhwan@andrew.cmu.edu

Please cite: 
Soft Gazetteers for Low-Resource Named Entity Recognition (ACL 2020)
https://www.aclweb.org/anthology/2020.acl-main.722
"""

from __future__ import print_function
import sys
import _dynet as dy
from collections import defaultdict
import argparse
import glob
from crf import CRFModule
from cnn import CNNModule
import re
import os
import time
from subprocess import Popen, PIPE
from args import return_argparser
import numpy as np
import random
import logging

UNK = "$unk"
DATA = ""
DROPOUT = 0.5
NUM_TYPES = 3
CNN_OUT_SIZE = 30
FEAT_OUT_SIZE = 128
OUTPUT_FOLDER = "outputs/"
MODELS_FOLDER = "models/"
DIGIT_RE = re.compile(r"\d")


class NERTagger(object):
    def __init__(
        self,
        embed_size,
        word_hidden_size,
        training_file,
        dev_file,
        test_file,
        batch_size,
        model_file,
        lstm_feats,
        crf_feats,
        autoencoder,
        train_features,
        dev_features,
        test_features,
        testing,
        restart,
        feat_func,
    ):
        self.crf_feats = crf_feats
        self.lstm_feats = lstm_feats
        self.autoencoder = autoencoder
        self.embed_size = embed_size
        self.word_hidden_size = word_hidden_size
        self.model_file = model_file

        self.featsize = 0

        self.word_vocab = defaultdict(lambda: len(self.word_vocab))
        self.char_vocab = defaultdict(lambda: len(self.char_vocab))
        self.tag_vocab = defaultdict(lambda: len(self.tag_vocab))
        self.word_lookup = []

        self.training_data = self.read_train(training_file, train_features)
        self.dev_data = self.read_test(dev_file, dev_features)
        self.test_data = self.read_test(test_file, test_features)
        self.batch_size = batch_size
        self.reverse_tag_lookup = dict((v, k) for k, v in self.tag_vocab.items())
        self.reverse_word_lookup = dict((v, k) for k, v in self.word_vocab.items())

        self.model = dy.ParameterCollection()

        self.cnn = CNNModule(self.model, self.char_vocab)
        self.word_embeds = self.model.add_lookup_parameters(
            (len(self.word_vocab), embed_size)
        )
        arr = np.array(self.word_lookup)
        self.word_embeds.init_from_array(arr)
        self.word_lstm = dy.BiRNNBuilder(
            1,
            CNN_OUT_SIZE + embed_size + FEAT_OUT_SIZE,
            word_hidden_size,
            self.model,
            dy.LSTMBuilder,
        )

        self.feat_w = self.model.add_parameters((FEAT_OUT_SIZE, self.featsize))
        self.feat_b = self.model.add_parameters((FEAT_OUT_SIZE))
        self.feat_func = feat_func

        num_tags = len(self.tag_vocab) + 2
        self.num_tags = num_tags

        # Last linear layer to map the output of the LSTM to the tag space
        self.context_to_emit_w = self.model.add_parameters(
            (len(self.tag_vocab), word_hidden_size + FEAT_OUT_SIZE)
        )
        self.context_to_emit_b = self.model.add_parameters((len(self.tag_vocab)))
        self.crf_module = CRFModule(self.model, self.tag_vocab)

        self.o_tag = self.tag_vocab["O"]

        self.context_to_trans_w = self.model.add_parameters(
            (num_tags * num_tags, word_hidden_size + FEAT_OUT_SIZE)
        )
        self.context_to_trans_b = self.model.add_parameters((num_tags * num_tags))

        self.feat_reconstruct_w = self.model.add_parameters(
            (self.featsize, word_hidden_size)
        )
        self.feat_reconstruct_b = self.model.add_parameters((self.featsize))

        if DROPOUT > 0.0:
            self.word_lstm.set_dropout(DROPOUT)

        if os.path.exists(self.model_file) and (testing or restart):
            self.model.populate(self.model_file)
            print("Populated!")
            v_acc = self.get_accuracy(self.dev_data, print_out="dev.")
            print("Validation F1: %f\n" % v_acc)

    def save_model(self):
        self.model.save(self.model_file)

    # Read the training data and builds the tag/word vocabulary
    def read_train(self, filename, feature_file):
        train_sents = []

        feature_vectors = np.load(feature_file, allow_pickle=True)["feats"]
        if self.featsize == 0:
            self.featsize = int(feature_vectors[0][0].shape[0])
        sent_count = 0
        with open(filename, "r", encoding="utf8") as fh:
            sent = []
            for line in fh:
                spl = line.strip().split()
                if line == "\n":
                    cur_feats = feature_vectors[sent_count]
                    sent = [
                        (
                            [self.char_vocab[c] for c in word],
                            self.word_to_int(word),
                            feats,
                            self.tag_vocab[tag],
                        )
                        for (word, tag), feats in zip(sent, cur_feats)
                    ]
                    train_sents.append(sent)
                    sent_count += 1
                    sent = []
                    continue
                sent.append((spl[0], spl[-1]))
        cur_feats = feature_vectors[sent_count]
        sent = [
            (
                [self.char_vocab[c] for c in word],
                self.word_to_int(word),
                feats,
                self.tag_vocab[tag],
            )
            for (word, tag), feats in zip(sent, cur_feats)
        ]
        train_sents.append(sent)

        return train_sents

    # Read the validation and test sets
    def read_test(self, filename, feature_file):
        sents = []

        feature_vectors = np.load(feature_file, allow_pickle=True)["feats"]
        sent_count = 0
        with open(filename, "r", encoding="utf8") as f:
            sent = []
            for line in f:
                spl = line.strip().split()
                if line == "\n":
                    cur_feats = feature_vectors[sent_count]
                    sent = [
                        (
                            [self.char_to_int(c) for c in word],
                            self.word_to_int(word),
                            feats,
                            self.tag_vocab[tag],
                        )
                        for (word, tag), feats in zip(sent, cur_feats)
                    ]
                    sents.append(sent)
                    sent = []
                    sent_count += 1
                    continue
                sent.append((spl[0], spl[-1]))
        cur_feats = feature_vectors[sent_count]
        sent = [
            (
                [self.char_to_int(c) for c in word],
                self.word_to_int(word),
                feats,
                self.tag_vocab[tag],
            )
            for (word, tag), feats in zip(sent, cur_feats)
        ]
        sents.append(sent)
        return sents

    # Get char ID
    def char_to_int(self, char):
        if char in self.char_vocab:
            return self.char_vocab[char]
        else:
            return self.char_vocab[UNK]

    def word_to_int(self, word):
        word = DIGIT_RE.sub("0", word)
        if word in self.word_vocab:
            return self.word_vocab[word]
        else:
            vec = np.random.uniform(
                low=-np.sqrt(3.0 / self.embed_size),
                high=np.sqrt(3.0 / self.embed_size),
                size=(self.embed_size,),
            )
            self.word_lookup.append(vec.tolist())
            return self.word_vocab[word]

    # Get tag ID
    def lookup_tag(self, tag_id):
        return self.reverse_tag_lookup[tag_id]

    def lookup_word(self, word_id):
        return self.reverse_word_lookup[word_id]

    # Get LSTM features in tag space for CRF decoding
    def get_features_for_tagging(self, sentence, training):
        word_feats = [
            dy.affine_transform(
                [
                    self.feat_b,
                    self.feat_w,
                    dy.inputTensor(feats.reshape(self.featsize, 1)),
                ]
            )
            for chars, word, feats, tag in sentence
        ]
        zero_feats = [
            dy.inputTensor(np.zeros(shape=(FEAT_OUT_SIZE, 1)))
            for chars, word, feats, tag in sentence
        ]

        # Non-linear transform for soft gazetteer features
        if self.feat_func == "tanh":
            word_feats = [dy.tanh(feat) for feat in word_feats]
        elif self.feat_func == "relu":
            word_feats = [dy.rectify(feat) for feat in word_feats]

        # Soft gazetteer features at the LSTM level
        if self.lstm_feats:
            cur_feats = word_feats
        else:
            cur_feats = zero_feats
        word_reps = [
            dy.concatenate(
                [self.cnn.encode(chars, training), self.word_embeds[word], enc_feat]
            )
            for enc_feat, (chars, word, feats, tag) in zip(cur_feats, sentence)
        ]

        contexts = self.word_lstm.transduce(word_reps)

        # Soft gazetteer features at the CRF level
        if self.crf_feats:
            cur_feats = word_feats
        else:
            cur_feats = zero_feats

        features = [
            dy.affine_transform(
                [
                    self.context_to_emit_b,
                    self.context_to_emit_w,
                    dy.concatenate([context, feats]),
                ]
            )
            for context, feats in zip(contexts, cur_feats)
        ]
        t_features = [
            dy.reshape(
                dy.affine_transform(
                    [
                        self.context_to_trans_b,
                        self.context_to_trans_w,
                        dy.concatenate([context, feats]),
                    ]
                ),
                (self.num_tags, self.num_tags),
            )
            for context, feats in zip(contexts, cur_feats)
        ]

        # Autoencoder feature reconstruction
        if self.lstm_feats:
            feat_reconstruct = [
                dy.logistic(
                    dy.affine_transform(
                        [self.feat_reconstruct_b, self.feat_reconstruct_w, context]
                    )
                )
                for context in contexts
            ]
        else:
            feat_reconstruct = [
                dy.inputTensor(np.zeros(shape=(self.featsize,))) for context in contexts
            ]

        return features, t_features, feat_reconstruct

    # Forward pass - BiLSTM + CRF, returns predicted tags obtained from viterbi decoding per sentence
    def get_output(self, sents):
        dy.renew_cg()
        tagged_sents = []
        for sent in sents:
            features, t_feats, _ = self.get_features_for_tagging(sent, False)
            cur_tag_seq, _ = self.crf_module.viterbi_decoding(features, t_feats)
            tagged_sents.append(cur_tag_seq)
        return tagged_sents

    # Calculate the loss - neg log likelihood from crf
    def calculate_loss(self, sents):
        dy.renew_cg()
        losses = []
        for sent in sents:
            features, t_features, feat_reconstruct = self.get_features_for_tagging(
                sent, True
            )
            gold_tags = [tag for chars, word, feats, tag in sent]
            cur_loss = self.crf_module.negative_log_loss(
                features, t_features, gold_tags
            )
            if self.autoencoder:
                autoencoder_loss = [
                    dy.binary_log_loss(reconstruct, dy.inputTensor(feats))
                    for reconstruct, (chars, word, feats, tag) in zip(
                        feat_reconstruct, sent
                    )
                ]
            else:  # remove autoencoder loss
                autoencoder_loss = [dy.scalarInput(0)]
            losses.append(cur_loss + (dy.esum(autoencoder_loss) / self.featsize))

        return dy.esum(losses)

    def train(self, epochs, trainer, lr, no_decay, patience, end_patience):
        if trainer == "sgd":
            trainer = dy.MomentumSGDTrainer(self.model, learning_rate=lr)
            trainer.set_clip_threshold(5.0)
        else:
            trainer = dy.AdamTrainer(self.model)
        best_acc = 0

        print(len(self.training_data))

        check_val = int(len(self.training_data) / (5.0 * self.batch_size))
        best_ep = -1
        for ep in range(epochs):
            logging.info("Epoch: %d" % ep)
            ep_loss = 0
            num_batches = 0
            random.shuffle(self.training_data)
            for i in range(0, len(self.training_data), self.batch_size):
                if num_batches % check_val == 0:
                    v_acc = self.get_accuracy(self.dev_data, print_out="dev.temp.")
                    logging.info("Validation F1: %f" % v_acc)

                    if v_acc > best_acc:
                        self.save_model()
                        best_acc = v_acc
                        logging.info("Saved!")
                        best_ep = ep
                cur_size = min(self.batch_size, len(self.training_data) - i)
                loss = self.calculate_loss(self.training_data[i : i + cur_size])
                ep_loss += loss.scalar_value()
                loss.backward()
                trainer.update()
                num_batches += 1
            logging.info("Training loss: %f" % ep_loss)
            if (ep - best_ep) > end_patience:
                self.model.populate(self.model_file)
                logging.info("Training patience reached.\n")
                break
            if not no_decay and (ep - best_ep) > patience:
                self.model.populate(self.model_file)
                # best_ep = ep
                lr = trainer.learning_rate / 1.05
                trainer.learning_rate = lr
                logging.info("New learning rate: " + str(lr))
            logging.info("\n")

    def get_accuracy(self, sents, print_out="temp."):
        if DROPOUT > 0.0:
            self.word_lstm.set_dropout(0)
        outputs = []
        for i in range(0, len(sents), self.batch_size):
            cur_size = min(self.batch_size, len(sents) - i)
            outputs += self.get_output(sents[i : i + cur_size])
        if DROPOUT > 0.0:
            self.word_lstm.set_dropout(DROPOUT)

        outfile = OUTPUT_FOLDER + print_out + output_name
        with open(OUTPUT_FOLDER + print_out + output_name, "w", encoding="utf8") as out:
            for sent, output in zip(sents, outputs):
                for (_, word, _, tag), pred_tag in zip(sent, output):
                    out.write(
                        self.lookup_word(word)
                        + " "
                        + self.lookup_tag(tag).upper()
                        + " "
                        + self.lookup_tag(pred_tag).upper()
                        + "\n"
                    )
                out.write("\n")
            out.write("\n")

        cmd = "cat " + outfile + " | " + "./conlleval_f1"
        process = Popen(cmd, stdout=PIPE, stdin=PIPE, shell=True)
        f1 = float(process.communicate()[0].strip())

        return f1


if __name__ == "__main__":
    parser = return_argparser()
    args, unknown = parser.parse_known_args()
    output_name = args.output_name

    tagger_model = NERTagger(
        embed_size=args.embed,
        word_hidden_size=args.word_hidden_size,
        training_file=args.train,
        dev_file=args.dev,
        test_file=args.test,
        batch_size=args.batch_size,
        train_features=args.train_feats,
        dev_features=args.dev_feats,
        test_features=args.test_feats,
        model_file=MODELS_FOLDER + output_name,
        lstm_feats=args.lstm_feats,
        crf_feats=args.crf_feats,
        autoencoder=args.autoencoder,
        testing=args.testing,
        restart=args.restart,
        feat_func=args.feat_func,
    )

    if int(args.testing):
        test_acc = tagger_model.get_accuracy(tagger_model.test_data, print_out="test.")
        print("Test F1: %f\n" % test_acc)
    else:
        logging.basicConfig(
            filename=OUTPUT_FOLDER + output_name + ".log",
            level=logging.INFO,
            format="%(message)s",
            filemode="w",
        )
        logging.info(str(args) + "\n")

        start = time.time()
        tagger_model.train(
            epochs=200,
            trainer=args.trainer,
            lr=args.lr,
            no_decay=args.no_decay,
            patience=5,
            end_patience=30,
        )
        end = time.time()
        logging.info("Training time: %0.4f\n" % (end - start))

        if args.test:
            test_acc = tagger_model.get_accuracy(
                tagger_model.test_data, print_out="test."
            )
            logging.info("Test F1: %f\n" % test_acc)
