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


def test(p, json_file, year, min_conf_score=0, jsonify=False, output_file="predictions.json"):
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
        #res = p.run(query=q, params={"retriever": {"top_k": 1, "filters":{ "name": [example['topic']], "year": [year]}}})
        res = p.run(query=q, params={"retriever": {"top_k": 1}})
        # add get json file format code
        res = dumper(res)
        answers = []
        topics = {}
        contexts = {}
        # go through every reader-predicted answer and store answer and topic
        for answer in res["answers"]:
            if answer["score"] >= min_conf_score:
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
        with open(output_file, 'w') as fp:
            json.dump(d, fp, default=dumper)

    return outputs
