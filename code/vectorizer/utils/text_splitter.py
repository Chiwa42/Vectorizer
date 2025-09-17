import os
import logging
from typing import List

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextSplitter:
    """
    負責文本分割處理。
    """
    
    def __init__(self, chunk_size=2048, chunk_overlap=100, min_chunk_size=50):
        """
        初始化文本分割器
        
        Args:
            chunk_size: 每個分塊的大小
            chunk_overlap: 分塊之間的重疊大小
            min_chunk_size: 觸發分割的最小文件大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split_text(self, text: str, source_path: str) -> List[Document]:
        """
        分割文本為多個文檔塊
        
        Args:
            text: 要分割的文本
            source_path: 源文件路徑
            
        Returns:
            分割後的文檔列表
        """
        filename = os.path.basename(source_path)
        
        if len(text) <= self.min_chunk_size:
            logging.info(f"文件 {filename} 長度 {len(text)} 字元，無需分割")
            return [Document(
                page_content=text,
                metadata={
                    "source": source_path,
                    "filename": filename,
                    "chunk_id": 0,
                    "total_chunks": 1,
                    "chunk_size": len(text)
                }
            )]
        
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source_path,
                    "filename": filename,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        logging.info(f"文件 {filename} 分割為 {len(chunks)} 個塊，每塊大小：{self.chunk_size}，重疊：{self.chunk_overlap}")
        
        return documents