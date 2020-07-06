"""
Get a list of ngrams from conll file(s).

Author: Shruti Rijhwani
Contact: srijhwan@andrew.cmu.edu

Please cite: 
Soft Gazetteers for Low-Resource Named Entity Recognition (ACL 2020)
https://www.aclweb.org/anthology/2020.acl-main.722
"""

import argparse


def ngrams_from_start_index(word_list, start_index, max_len):
    ngrams = {}
    for i in range(start_index + 1, min(start_index + max_len + 1, len(word_list) + 1)):
        ngrams[" ".join(word_list[start_index:i])] = True
    return ngrams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filenames", nargs="+", help="conll files to extract ngrams from"
    )
    parser.add_argument(
        "--n", default=3, type=int, help="maximum length of ngrams to extract"
    )
    parser.add_argument("--output", help="output filename for list of ngrams")
    args = parser.parse_args()

    ngrams = {}
    sent = []
    max_len = args.n

    for filen in args.filenames:
        with open(filen, "r", encoding="utf8") as f:
            for line in f:
                if line == "\n":
                    for i in range(0, len(sent)):
                        ngrams.update(ngrams_from_start_index(sent, i, max_len))
                    sent = []
                    continue
                spl = line.strip().split()
                sent.append(spl[0])
        for i in range(0, len(sent)):
            ngrams.update(ngrams_from_start_index(sent, i, max_len))
        sent = []

    with open(args.output, "w", encoding="utf8") as out:
        for k, v in ngrams.items():
            out.write(k + "\n")
