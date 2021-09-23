import gzip
from tqdm import tqdm
import json
import sys
import os
import ast




def get_haystack_format(directory):
    i = 0
    documents = []

    for file in os.listdir(directory):
        with open(os.path.join(directory, file)) as f:
            print(f"Parsing {file}")
            for line in tqdm(f):
                parsed = ast.literal_eval(line)
                if 'text' in parsed:
                    if parsed['text'] != '':
                        documents.append(parsed)
                        i += 1

    return documents
    
'''
Usage: python <process.py> <directory with processed dump files>
'''
if __name__ == "__main__":
    directory = sys.argv[1]
    docs = get_haystack_format(directory)
    print(f"Found {len(docs)} non empty documents in all files.")
