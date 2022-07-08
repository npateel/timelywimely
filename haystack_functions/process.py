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


#documents = []

def dumper(obj):
    try:
        return obj.to_dict()
    except:
        return obj.__dict__

def get_haystack_format(directory, out):

    filenames = []
    for filename in glob.iglob(directory + '/**/*', recursive=True):
        if os.path.isfile(filename):
            filenames.append(filename)
    print(len(filenames))


    haystack_file = open(out, mode='w')

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
                    json.dump(d, out)
                    out.write('\n')
    #documents.extend(docs)



'''
Usage: python <process.py> <directory with processed dump files>
'''

def processDump(year):
    directory = year
    get_haystack_format(os.path.join(directory, 'dump'), os.path.join(directory, 'dump.jsonl'))
    print(f"Wrote non empty docs to dump")
    url = "sqlite:///" + os.path.join(directory, 'faiss_document_store.db')
    document_store = FAISSDocumentStore(sql_url=url,progress_bar=True,faiss_index_factory_str="Flat")
    i = 0
    docs= []
    with open(os.path.join(directory, 'dump.jsonl')) as f:
        for line in tqdm(f):
            i += 1
            docs.append(json.loads(line))
            if i == 1000:
                document_store.write_documents(docs)
                docs = []
                i = 0
    print("Done creating documents")

    document_store.save(os.path.join(directory, "wiki_dump.faiss"))
    print("Saved document store")
    document_store = FAISSDocumentStore.load(os.path.join(directory, "wiki_dump.faiss"), sql_url=url, index='document')
    print("Starting DPR")
    retriever = DensePassageRetriever(document_store=document_store, query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                      passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                      use_gpu=True, batch_size=64, embed_title=True)
    print("Updating embeddings...'")
    document_store.update_embeddings(retriever)
    document_store.save(os.path.join(directory, "wiki_dump_embeddings.faiss"))





if __name__ == "__main__":
    directory = sys.argv[1]
    processDump(directory)
    sys.exit()


    get_haystack_format(directory)
    print(f"Found {len(documents)} non empty documents in all files.")

    #document_store = FAISSDocumentStore(similarity="dot_product")
    document_store = FAISSDocumentStore(sql_url="sqlite:///faiss_document_store.db",progress_bar=True,faiss_index_factory_str="Flat")
    #filename = sys.argv[1]
    i = 0
    docs = []
    #with open(filename) as f:
     #   for line in tqdm(f):
      #      i+= 1
       #     docs.append(json.loads(line))
        #    if i == 10000:
         #       document_store.write_documents(docs)
          #      docs = []
           #     i= 0


    #print("Done creating documents")

    #document_store.save("wiki_dump.faiss")
    #print("Saved document store")
    

    # create document store from the documents
    # I think this is more optimized for DPR?

    # faiss_index_factory_str is needed for saving and loading to work properly
    document_store = FAISSDocumentStore.load("wiki_dump.faiss", sql_url='sqlite:///faiss_document_store.db', index='document')
    # unsure if we should tune these parameters based on Nikhils gpu
    print("Starting DPR")
    retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  use_gpu=True,
                                  batch_size=256,
                                  embed_title=True)
    # apparently this is time consuming
    print("Updating embeddings...'")
    document_store.update_embeddings(retriever)
    document_store.save("wiki_dump_embeddings.faiss")
    # example:
    # retrieved_doc = retriever.retrieve(query="Why did the revenue increase?")

    # tune these parameters too
    print("Running QA Reader")
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2",
                use_gpu=True, no_ans_boost=-10, context_window_size=500,
                top_k_per_candidate=3, top_k_per_sample=1,
                num_processes=1, max_seq_len=256, doc_stride=128)

    # example:
    # reader.predict(question="Who is the father of Arya Starck?", documents=retrieved_docs, top_k=3)

    print("Started pipeline")
    p = Pipeline()
    p.add_node(component=retriever, name="ESRetriever1", inputs=["Query"])
    p.add_node(component=reader, name="QAReader", inputs=["ESRetriever1"])
    res = p.run(query="What did Einstein work on?", params={"retriever": {"top_k": 1}})
    json.dump(res, open('answer.json', 'w'), default=dumper) 
    exit()
