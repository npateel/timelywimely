import gzip
from tqdm import tqdm
import json
import sys
import os
from multiprocessing import Pool
import ast
import glob
from haystack.document_store import FAISSDocumentStore
from haystack.retriever.dense import DensePassageRetriever
from haystack.reader import FARMReader
from haystack import Pipeline

def test(p, json_file, max_ans = 1, min_conf_score = .7, jsonify = false):
    # p: haystack Pipeline
    # json_file: take a wild guess, lol just name

    # combine path
    json_path = os.path.join("../datasets", json_file)

    # open json file into dict
    with open(json_path) as jf:
        dataset = json.load(jf)

    outputs = []
    # go through all qs and store
    for example in dataset["data"]:
        output = {}
        q = example["question"]
        res = p.run(query=q, params={"retriever": {"top_k": 1}})
        # add get json file format code
        # res = smth
        answers = []
        topics = {}
        # go through every reader-predicted answer and store answer and topic
        for answer in res["answers"]:
            if answer["score"] >= min_conf_score and len(answers) < max_ans:
                answers.append(answer)
                topics[answer] = answer["meta"]["name"]

        output["question"] = q
        output["gt_ans"] = example["answers"]
        output["predicted_ans"] = answers
        output["gt_topic"] = example["topic"]
        output["predicted_topic"] = topics

        outputs.append(output)

    if jsonify:
        d = {"data": outputs}
        with open('predictions.json', 'w') as fp:
            json.dump(d, fp)

    return outputs
