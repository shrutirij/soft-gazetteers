"""
Convert the .tab format from LDC to conll format for the LORELEI dataset.

Author: Shruti Rijhwani
Contact: srijhwan@andrew.cmu.edu

Please cite: 
Soft Gazetteers for Low-Resource Named Entity Recognition (ACL 2020)
https://www.aclweb.org/anthology/2020.acl-main.722
"""

from collections import defaultdict
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tac", help=".tab file of gold edl annotations")
    parser.add_argument("--conll", help="ldc conll format")
    parser.add_argument("--out", help="output filename")
    parser.add_argument("--filelist", help="list of annotated files")
    args = parser.parse_args()

    filelist = args.filelist
    tac_file = args.tac
    conll_file = args.conll
    output = args.out

    doc = {}
    doc_ids = {}

    with open(filelist, "r") as f:
        for line in f:
            doc_ids[line.strip()] = True

    docx = {}

    with open(tac_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            tokens = line.split("\t")
            doc_id = tokens[3].split(":")[0]
            start = tokens[3].split(":")[1].split("-")[0]
            end = tokens[3].split(":")[1].split("-")[1]
            tag = tokens[5]
            if doc_id not in doc_ids:
                docx[doc_id] = True
            else:
                doc_ids[doc_id] = False
            doc[doc_id + "@@" + start] = end + "@@" + tag + "@@" + tokens[4]

    fout = open(output, "w", encoding="utf-8")

    # Pruned filelist of annotations that exist in the gold file
    with open(filelist + ".clean", "w") as f:
        for doc_id, val in doc_ids.items():
            if not val:
                f.write(doc_id + "\n")

    with open(conll_file, "r", encoding="utf-8") as fin:
        inEntity = False
        tag = ""
        end_pos = -1
        cur_doc = ""
        present = False

        for line in fin:
            line = line.strip()

            if line == "" or line == "\n":
                if present:
                    fout.write("\n")
            else:
                tokens = line.split()
                assert len(tokens) == 10 or len(tokens) == 11
                token = tokens[0]
                doc_id = tokens[3]
                start = tokens[6]
                end = tokens[7]
                if doc_id not in doc_ids:
                    present = False
                    continue

                if doc_ids[doc_id] == True:
                    present = False
                    # print(doc_id)
                    continue

                present = True
                key = doc_id + "@@" + start

                if key in doc:
                    end_pos = doc[key].split("@@")[0]
                    tag = doc[key].split("@@")[1]
                    fout.write(token + " " + doc_id + " " + "B-" + tag + "\n")
                    inEntity = True
                    cur_doc = doc_id

                elif inEntity and end_pos != -1 and tag != "" and doc_id == cur_doc:
                    if int(end) <= int(end_pos):
                        fout.write(token + " " + doc_id + " " + "I-" + tag + "\n")
                    else:
                        fout.write(token + " " + doc_id + " " + "O" + "\n")
                        inEntity = False
                        tag = ""
                        end_pos = -1

                else:
                    fout.write(token + " " + doc_id + " " + "O" + "\n")
