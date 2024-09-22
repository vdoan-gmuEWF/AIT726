from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pacmap
import numpy as np
import plotly.express as px

import warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

class Retriever:
    def __init__(self, file_path: str):
        self.file_path = file_path
        loader = CSVLoader(file_path=file_path, csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['Name', 'Text']
        })
        self.RAW_KNOWLEDGE_BASE = loader.load()
        self.MARKDOWN_SEPARATORS = [
                                    "\n#{1,6} ",
                                    "```\n",
                                    "\n\\*\\*\\*+\n",
                                    "\n---+\n",
                                    "\n___+\n",
                                    "\n\n",
                                    "\n",
                                    " ",
                                    "",
                                    ]
        self.EMBEDDING_MODEL_NAME = "thenlper/gte-small"
        self.KNOWLEDGE_VECTOR_DATABASE = self.create_vector_database()

    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME),
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS
        )
        docs_processed = []
        current_row = None
        chunk_index = 0

        for document in self.RAW_KNOWLEDGE_BASE:
            if document.metadata['row'] != current_row:
                current_row = document.metadata['row']
                chunk_index = 1
            else:
                chunk_index += 1
            document.metadata["id"] = f"{document.metadata.get('source', 'unknown')} :{document.metadata['row']}:{chunk_index}"
            docs_processed += text_splitter.split_documents([document])
            
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique
            
    
    def transform(self):
        print(f"Model's maximum sequence length: {SentenceTransformer('thenlper/gte-small').max_seq_length}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.EMBEDDING_MODEL_NAME)
        lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(self.split_documents())]

        # Plot the distribution of document lengths, counted as the number of tokens
        fig = pd.Series(lengths).hist()
        plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
        plt.show()
        
    def get_embedding_model(self):
        try:
            return HuggingFaceEmbeddings(
                                            model_name=self.EMBEDDING_MODEL_NAME,
                                            multi_process=True,
                                            encode_kwargs={"normalize_embeddings": True}
                                        )
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return None
        
    def create_vector_database(self):
        return FAISS.from_documents(self.split_documents(), self.get_embedding_model(), distance_strategy=DistanceStrategy.COSINE)
    
    def project_2d_embedding(self, user_query):
        embedding_model = self.get_embedding_model()
        query_vector = embedding_model.embed_query(user_query)
        docs_processed = self.split_documents()
        embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)
        embeddings_2d = [
            list(self.KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
        ] + [query_vector]
        documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")
        
        df = pd.DataFrame.from_dict(
            [
                {
                    "x": documents_projected[i, 0],
                    "y": documents_projected[i, 1],
                    "source": docs_processed[i].metadata["source"].split("/")[1],
                    "extract": docs_processed[i].page_content[:100] + "...",
                    "symbol": "circle",
                    "size_col": 4,
                }
                for i in range(len(docs_processed))
            ]
            + [
                {
                    "x": documents_projected[-1, 0],
                    "y": documents_projected[-1, 1],
                    "source": "User query",
                    "extract": user_query,
                    "size_col": 100,
                    "symbol": "star",
                }
            ]
        )

        # Visualize the embedding
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="source",
            hover_data="extract",
            size="size_col",
            symbol="symbol",
            color_discrete_map={"User query": "black"},
            width=1000,
            height=700,
        )
        fig.update_traces(
            marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
            selector=dict(mode="markers"),
        )
        fig.update_layout(
            legend_title_text="<b>Chunk source</b>",
            title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
        )
        fig.show()
            
    def retrive(self, user_query):
        return self.KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)