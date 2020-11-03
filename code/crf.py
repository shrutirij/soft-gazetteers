"""
Implements a CRF for sequential decoding of NER tags for a sentence.

Author: Shruti Rijhwani and Samridhi Shree Choudhary
Contact: srijhwan@andrew.cmu.edu

Please cite: 
Soft Gazetteers for Low-Resource Named Entity Recognition (ACL 2020)
https://www.aclweb.org/anthology/2020.acl-main.722
"""

import dynet as dy
import numpy as np

CONST_INIT = -1e10


class CRFModule:
    def __init__(self, model, tag_vocab):
        self.tag_vocab = tag_vocab
        self.tag_lookup = dict((v, k) for k, v in tag_vocab.items())
        self.begin_tag = len(tag_vocab)
        self.end_tag = len(tag_vocab) + 1
        self.num_tags = len(tag_vocab) + 2

        # Matrix of transition parameters. T[i,j] = score of transitioning from tag_i to tag_j
        self.transitions = model.add_lookup_parameters((self.num_tags, self.num_tags))

    """
    Numerically stable log-sum-exp for forward computation
    """

    def log_sum_exp(self, scores):
        max_score = dy.pick(scores, np.argmax(scores.npvalue()))  # dynet expression
        max_score_broadcast = dy.concatenate([max_score] * self.num_tags)
        # print(scores.npvalue().shape)
        # print(max_score_broadcast)
        return max_score + dy.logsumexp_dim((scores - max_score_broadcast))

    def log_sum_exp_matrix(self, scores):
        max_scores = dy.inputVector(np.max(scores.npvalue(), 1))
        temp_expr = dy.logsumexp_dim(dy.transpose(scores - max_scores))
        return max_scores + temp_expr

    """
    Forward algorithm for calculating the partition function
    """

    def forward(self, features, t_features):
        init_alphas = [CONST_INIT] * self.num_tags
        # Scores for start tag is zero
        init_alphas[self.begin_tag] = 0
        alpha_expr = dy.inputVector(init_alphas)

        # Keep updating the expression at each time step to interatively calculate the partition function
        for i, feature in enumerate(features):
            temp_expr = dy.transpose(alpha_expr) + (t_features[i] + feature)
            alpha_expr = self.log_sum_exp_matrix(temp_expr)

        last_transition = alpha_expr + t_features[-1][self.end_tag]
        return self.log_sum_exp(last_transition)

    """
    Calculates the score of a tag sequence. 
    Score(x,y) = sum(log emit(xi->yi) + log trans(y_i-1->y_i)) = sum(h_i[y_i] + T[y_i,y_i-1])
    h_i[y_i] = output state of the LSTM
    Starting from begin tag, so i->i+1
    """

    def score_sentence(self, features, t_features, tags):
        score = dy.scalarInput(0)
        tags = [self.begin_tag] + tags

        for i, feat in enumerate(features):
            score = (
                score
                + dy.pick(t_features[i][tags[i + 1]], tags[i])
                + dy.pick(feat, tags[i + 1])
            )

        # Last transition to end tag from last tag
        score = score + dy.pick(t_features[-1][self.end_tag], tags[-1])
        return score

    """
    Viterbi Decoding for optimal tag sequence.
    Also returns the score for the optimal path
    """

    def viterbi_decoding(self, features, t_features):
        back_pointers = []
        init_scores = [CONST_INIT] * self.num_tags
        # Start tag has zero score
        init_scores[self.begin_tag] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = dy.inputVector(init_scores)

        for i, feat in enumerate(features):
            # holds the backpointers for this step
            backpointers_t = []
            # holds viterbi scores at this time step
            viterbivars_t = []

            next_tag_expr = dy.transpose(forward_var) + t_features[i]
            backpointers_t = np.argmax(next_tag_expr.npvalue(), 1)
            viterbivars_t = dy.inputVector(np.max(next_tag_expr.npvalue(), 1))

            # Add the emission score - feat
            feat = dy.concatenate([feat, dy.inputVector([CONST_INIT, CONST_INIT])], d=0)
            forward_var = viterbivars_t + feat
            back_pointers.append(backpointers_t)

        # Transition to end tag and get the path score
        last_tag_expr = forward_var + t_features[-1][self.end_tag]
        best_id = np.argmax(last_tag_expr.npvalue())
        # print(type(best_id))
        path_score = dy.pick(last_tag_expr, best_id)

        # Follow the backpointers to decode the best path and calculate the score
        best_path = [best_id]
        for bptr in reversed(back_pointers):
            best_id = bptr[best_id]
            # print(type(best_id))
            best_path.append(best_id)
        # Remove the start symbol as we do not want to return that to the classifier
        best_path.pop()
        best_path.reverse()
        return best_path, path_score

    """
    Calculates the negative log loss (forward score - likelihood, sentence score - true score)
    """

    def negative_log_loss(self, features, t_features, tags):
        # add the begin and end tag init scores for each of the features
        features = [
            dy.concatenate([feat, dy.inputVector([CONST_INIT, CONST_INIT])], d=0)
            for feat in features
        ]
        forward_score = self.forward(features, t_features)
        gold_score = self.score_sentence(features, t_features, tags)
        return forward_score - gold_score
