#!/bin/python3
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
    documents = get_haystack_format(directory)
    print(f"Found {len(documents)} non empty documents in all files.")

    # create document store from the documents
    # I think this is more optimized for DPR?
    # document_store = FAISSDocumentStore(similarity="dot_product")

    # faiss_index_factory_str is needed for saving and loading to work properly
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    document_store.write_documents(documents)
    # document_store.save("wiki_dump.faiss")
    # document_store = FAISSDocumentStore.load("wiki_dump.faiss")

    # unsure if we should tune these parameters based on Nikhils gpu
    retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  use_gpu=True,
                                  batch_size=16,
                                  embed_title=True)
    # apparently this is time consuming
    document_store.update_embeddings(retriever)
    # example:
    # retrieved_doc = retriever.retrieve(query="Why did the revenue increase?")

    # tune these parameters too
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                use_gpu=False, no_ans_boost=-10, context_window_size=500,
                top_k_per_candidate=3, top_k_per_sample=1,
                num_processes=8, max_seq_len=256, doc_stride=128)

    # example:
    # reader.predict(question="Who is the father of Arya Starck?", documents=retrieved_docs, top_k=3)


    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever1", inputs=["Query"])
    p.add_node(component=reader, name="QAReader", inputs=["ESRetriever1"])
    res = p.run(query="What did Einstein work on?", params={"retriever": {"top_k": 1}})
