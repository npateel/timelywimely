import ast
import glob
import gzip
import json
import os
import sys
from multiprocessing import Pool

import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from haystack import Pipeline
from haystack.document_store import FAISSDocumentStore
from haystack.reader import FARMReader
from haystack.retriever.dense import DensePassageRetriever


def dumper(obj):
    try:
        return obj.to_dict()
    except:
        try:
            return obj.__dict__
        except:
            return obj


def test(p, json_file, max_ans=1, min_conf_score=.7, jsonify=False):
    # p: haystack Pipeline
    # json_file: take a wild guess, lol just name

    # combine path
    json_path = os.path.join("../datasets", json_file)

    # open json file into dict
    with open(json_path) as jf:
        dataset = json.load(jf)

    outputs = {}
    # go through all qs and store

    for example in tqdm(dataset["data"]):
        output = {}
        q = example["question"]
        res = p.run(query=q, params={"retriever": {"top_k": 1}})
        # add get json file format code
        res = dumper(res)
        answers = []
        topics = {}
        contexts = {}
        # go through every reader-predicted answer and store answer and topic
        for answer in res["answers"]:
            if answer["score"] >= min_conf_score and len(answers) < max_ans:
                answers.append(answer['answer'])
                topics[answer['answer']] = answer["meta"]["name"]
                contexts[answer['answer']] = answer['context']

        # output["question"] = q
        output["gt_ans"] = example["answers"]
        # think about how answer could be just a string
        output["predicted_ans"] = answers
        output["gt_topic"] = example["topic"]
        output["predicted_topic"] = topics
        output["predicted_context"] = contexts

        outputs[q] = output

    if jsonify:
        d = outputs
        with open('2016/predictions.json', 'w') as fp:
            json.dump(d, fp, default=dumper)

    return outputs


def dfOverTime(all_outputs, years, topic=True):
    # allOutputs: a list of all the outputs that come from test() for all years
    # years: a list of all the dates in the same order they're put in allOutputs
    #       for the purpose of naming columns
    # topic: boolean choosing whether you include topic columns

    # create empty dataframe all cols
    bleu_cols = [x + " Bleu Score"
                 for x in years[1:]]  # don't do bleu for first
    year_ans = [x + " Predicted" for x in years]
    if topic:
        year_topic = [x + " Topic" for x in years]
        year_ans = year_ans + year_topic
    sorted_cols = (year_ans + bleu_cols).sort()
    all_cols = ["Question", "GT Answer", "GT Topic"] + sorted_cols
    df = pd.DataFrame(columns=all_cols)

    # use the first dict as the reference
    base_output = all_outputs[0]

    # create a dictionary to hold lists of all cols
    acc_col_vals = dict.fromkeys(all_cols, [])

    # populate vals of each col one row at a time
    for q, rest in base_output.items():
        acc_col_vals["Question"].append(q)
        acc_col_vals["GT Answer"].append(rest["gt_ans"])
        acc_col_vals["GT Topic"].append(rest["gt_topic"])
        col_name = years[0] + " Predicted"
        # rest["predicted_ans"] is a list, append a string to the predicted
        # answer col
        ans_str = ""
        for ans in rest["predicted_ans"]:
            ans_str = ans_str + ans + ", "
        acc_col_vals[col_name].append(ans_str)
        # add all first year predicted answers to ref for bleu, is this ok?
        ref = [a.split(" ") for a in rest["predicted_ans"]]
        # for the rest of the years add answer and bleu score
        for i in range(1, len(years)):
            year = years[i]
            col_ans = year + " Predicted"
            col_bleu = year + " Bleu Score"
            all_ans = all_outputs[i][q]["predicted_ans"]
            if topic:
                col_top = year + " Topic"
                acc_col_vals[col_top] = all_outputs[i][q]["predicted_topic"]
            max_bleu = -1
            best_ans = "None"
            # choose the best answer according to unigram bleu
            for ans in all_ans:
                curr_bleu = 100 * sentence_bleu(
                    ref, ans.split(" "), weights=(1, 0, 0, 0))
                if curr_bleu > max_bleu:
                    max_bleu = curr_bleu
                    best_ans = ans
            acc_col_vals[col_ans].append(best_ans)
            acc_col_vals[col_bleu].append(max_bleu)
            ref.append(best_ans.split(" "))

    for col_name, col_data in acc_col_vals.items():
        df[col_name] = col_data

    return df
