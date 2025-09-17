import os
import logging
from typing import List

# 設定PyTorch記憶體優化環境變數
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma

class VectorDBManager:
    """
    負責向量資料庫的操作。
    """

    def __init__(self, persist_directory: str,
                 embedding_model_name: str = "BAAI/bge-m3",
                 model_kwargs: dict = {'device': 'cuda:0'},
                 encode_kwargs: dict = {'normalize_embeddings': True},
                 query_instruction = "為這個句子生成表示以用於檢索相關文章："):

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction=query_instruction
        )
        self.query_instruction = "為這個句子生成表示以用於檢索相關文章："
        self.persist_directory = persist_directory
        self.vectordb = self.load_db()

    def load_db(self) -> Chroma:
        """
        載入或創建向量資料庫。
        """
        try:
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        except ValueError:
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            vectordb.persist()
        
        import torch
        torch.cuda.empty_cache()
        
        return vectordb

    def add_documents(self, documents: List[Document]):
        """
        將文件新增到向量資料庫。
        """
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.vectordb.add_documents(batch)
            logging.info(f"已處理第 {i+1}-{min(i+batch_size, len(documents))} 個文檔塊")
            import torch
            torch.cuda.empty_cache()
        
        logging.info(f"成功將 {len(documents)} 個文檔塊加入向量資料庫")