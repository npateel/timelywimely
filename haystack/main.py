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

import process
import testing
from haystack import Pipeline
from haystack.document_store import FAISSDocumentStore, SQLDocumentStore
from haystack.reader import FARMReader
from haystack.retriever.sparse import TfidfRetriever
from haystack.retriever.dense import DensePassageRetriever, EmbeddingRetriever


class TopicDPR(DensePassageRetriever):
    def retrieve(self, query: str, filters: dict = None, top_k= None, index: str = None, **kwargs):
        topic = kwargs.get('topic') or None
        return super().retrieve(query=topic, filters=filters, top_k=top_k, index=index)




if __name__ == "__main__":
    year = sys.argv[1]
    dataset = sys.argv[2]
    output_path = sys.argv[3]
    faiss = os.path.join(year, 'wiki_dump_embeddings.faiss')
    sql_url = 'sqlite:///' + os.path.join(year, 'faiss_document_store.db')

    document_store = FAISSDocumentStore.load(faiss, sql_url=sql_url, index='document')
    #document_store = SQLDocumentStore(url=sql_url, index='document')
    print("Loaded document store")
    retriever = DensePassageRetriever( document_store=document_store, query_embedding_model="facebook/dpr-question_encoder-single-nq-base", passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base", use_gpu=True, batch_size=64, embed_title=True)
    #retriever = TfidfRetriever(document_store)
    print("Loaded retriever")
    reader = FARMReader(model_name_or_path="nlpconnect/roberta-base-squad2-nq",
                        use_gpu=True,
                        no_ans_boost=-10,
                        context_window_size=500,
                        top_k_per_candidate=3,
                        top_k_per_sample=1,
                        num_processes=1,
                        max_seq_len=256,
                        doc_stride=128,
                        progress_bar=False)
    print("Loaded reader")

    p = Pipeline()
    p.add_node(component=retriever, name="retriever", inputs=["Query"])
    p.add_node(component=reader, name="QAReader", inputs=["retriever"])
    testing.test(p, dataset, year, jsonify=True, output_file=output_path)
