import logging
from typing import List, Set
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorDBManager:
    """
    負責管理向量資料庫的類別，包括初始化、添加文件和查詢。
    """
    def __init__(self, db_path: str, model_name: str = "BAAI/bge-m3"):
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_model = self._initialize_embedding_model()
        self.vector_db = self._initialize_vector_db()

    def _initialize_embedding_model(self):
        """初始化嵌入模型，使用BAAI/bge-m3。"""
        logging.info(f"正在載入嵌入模型: {self.model_name}")
        try:
            model_kwargs = {'device': 'cuda'}
            encode_kwargs = {'normalize_embeddings': True}
            embedding_model = HuggingFaceBgeEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logging.info("嵌入模型載入成功。")
            return embedding_model
        except Exception as e:
            logging.error(f"載入嵌入模型時發生錯誤: {e}")
            raise

    def _initialize_vector_db(self):
        """初始化或連接到向量資料庫。"""
        logging.info(f"正在連接到向量資料庫，路徑為: {self.db_path}")
        try:
            vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
            )
            logging.info("向量資料庫連接成功。")
            return vector_db
        except Exception as e:
            logging.error(f"初始化向量資料庫時發生錯誤: {e}")
            raise

    def add_documents(self, documents: List[Document]):
        """將文件列表添加到向量資料庫。"""
        if not documents:
            logging.warning("文件列表為空，沒有文件需要添加。")
            return
        
        logging.info(f"正在將 {len(documents)} 個文件塊添加到向量資料庫。")
        # 移除多餘的括號
        self.vector_db.add_documents(documents)
        logging.info("文件添加完成並已保存。")
        
    def get_processed_paths(self) -> Set[str]:
        """
        從向量資料庫中獲取所有已處理過的文件路徑。
        這個方法會透過 metadata 查詢，以避免重複處理相同的文件。
        """
        try:
            # 獲取資料庫中的所有文件ID和metadata
            # ChromaDB 的 get() 方法如果沒有指定 ids 或 where 參數，會返回所有資料
            all_data = self.vector_db.get(include=['metadatas'])
            metadatas = all_data.get('metadatas', [])
            
            # 從 metadata 中提取文件路徑並存入集合以確保唯一性
            # 這裡假設文件路徑儲存在 metadata 的 'source' 欄位中
            paths = {meta.get('source') for meta in metadatas if 'source' in meta}
            
            return paths
        except Exception as e:
            logging.error(f"從向量資料庫中獲取已處理路徑時發生錯誤: {e}", exc_info=True)
            return set()

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        """執行向量查詢。"""
        return self.vector_db.similarity_search(query_text, k=k)