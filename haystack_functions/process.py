#!/bin/python3
import gzip
from tqdm import tqdm
import json
import sys
import os
from multiprocessing import Pool
import ast
import glob


documents = []

def get_haystack_format(directory):

    filenames = []
    for filename in glob.iglob(directory + '/**/*', recursive=True):
        if os.path.isfile(filename):
            filenames.append(filename)
    print(len(filenames))


    haystack_file = open('dump.out.jsonl', mode='w')

    documents = list(map(lambda x: processFile(x, haystack_file), filenames))


def processFile(filename, out):
    docs = []
    with open(filename) as f:
        for line in f:
            parsed = json.loads(line)
            if 'text' in parsed:
                if parsed['text'] != '':
                    d = {
                    'text': parsed['text'],
                    'meta': {'name': parsed['title'],
                            'url': parsed['url']}}
                    docs.append(d)
    documents.extend(docs)

'''
Usage: python <process.py> <directory with processed dump files>
'''
if __name__ == "__main__":
    directory = sys.argv[1]
    get_haystack_format(directory)
    print(f"Found {len(documents)} non empty documents in all files.")
