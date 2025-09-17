import os
import sys
import logging
import argparse
from typing import List, Dict, Any
from datetime import datetime
import json

# Hugging Face和向量資料庫相關
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

# 導入 Reranker 相關庫
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorQuerySystem:
    """
    向量資料庫查詢系統，增加 Reranker 進行二次排序
    """
    
    def __init__(self, 
                 chroma_db_dir: str = "../vector_dbs/dev_dbs/",
                 embedding_model_name: str = "BAAI/bge-m3",
                 reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
                 model_kwargs: dict = {'device': 'cuda:0'},
                 encode_kwargs: dict = {'normalize_embeddings': True},
                 query_instruction = "為這個句子生成表示以用於檢索相關文章："):
        """
        初始化查詢系統
        
        Args:
            chroma_db_dir: 向量資料庫目錄路徑
            embedding_model_name: 嵌入模型路徑或名稱
            reranker_model_name: Reranker 模型路徑或名稱
            model_kwargs: 模型參數
            encode_kwargs: 編碼參數
        """
        self.chroma_db_dir = chroma_db_dir
        self.tokenizer = None
        self.reranker_model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 檢查資料庫是否存在
        if not os.path.exists(chroma_db_dir):
            raise FileNotFoundError(f"向量資料庫目錄不存在: {chroma_db_dir}")

        # 初始化嵌入模型（必須使用與建立資料庫時相同的模型）
        try:
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction=query_instruction
            )
            logging.info("✅ 嵌入模型載入成功")
        except Exception as e:
            logging.error(f"❌ 嵌入模型載入失敗: {e}")
            raise
            
        # 載入 Reranker 模型
        try:
            logging.info(f"🚀 正在載入 Reranker 模型: {reranker_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
            self.reranker_model.to(self.device).eval()
            logging.info("✅ Reranker 模型載入成功")
        except Exception as e:
            logging.error(f"❌ Reranker 模型載入失敗: {e}")
            self.reranker_model = None
            logging.warning("⚠️ Reranker 模型載入失敗，將退回單純向量搜索模式")
            
        # 載入向量資料庫
        try:
            self.vectordb = Chroma(
                persist_directory=chroma_db_dir,
                embedding_function=self.embeddings
            )
            
            collection_count = self.vectordb._collection.count()
            if collection_count == 0:
                logging.warning("⚠️ 向量資料庫是空的，請先執行文檔向量化")
            else:
                logging.info(f"✅ 向量資料庫載入成功，包含 {collection_count} 個文檔片段")
                
        except Exception as e:
            logging.error(f"❌ 向量資料庫載入失敗: {e}")
            raise

    def search(self, query: str, top_k: int = 100, initial_k: int = 1000) -> List[Dict[str, Any]]:
        """
        執行雙階段語義搜索
        
        Args:
            query: 查詢問句或關鍵字
            top_k: 最終返回的結果數量（重排序後）
            initial_k: 初始向量搜索的候選集數量（重排序前）
            
        Returns:
            搜索結果列表
        """
        try:
            logging.info(f"🔍 執行搜索查詢: '{query}'")
            
            # --- 第一階段：向量相似度搜索 (Retrieval) ---
            initial_results = self.vectordb.similarity_search_with_score(query, k=initial_k)
            
            # 處理結果
            candidates = []
            seen_content = set()
            for doc, score in initial_results:
                content_hash = hash(doc.page_content[:100])
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                candidates.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'distance_score': score
                })
            
            if not candidates:
                logging.warning("⚠️ 向量搜索未找到任何結果")
                return []
                
            logging.info(f"✨ 第一階段找到 {len(candidates)} 個候選結果")

            # --- 第二階段：重排序 (Reranking) ---
            if self.reranker_model:
                # 將查詢與候選文檔內容配對
                pairs = [[query, candidate['content']] for candidate in candidates]

                # 使用 Reranker 進行二次評分
                with torch.no_grad():
                    inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()

                # 將分數加入到候選結果中
                for i, score in enumerate(scores):
                    candidates[i]['rerank_score'] = score.item()
                
                # 根據 Reranker 分數進行排序（分數越高越好）
                candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
                
                logging.info("✅ Reranker 重排序成功")
            else:
                # 如果 Reranker 載入失敗，則退回使用初始的距離分數排序
                candidates.sort(key=lambda x: x['distance_score'])
                logging.warning("⚠️ 由於 Reranker 載入失敗，結果使用初始距離分數排序")

            # 取出最終的 top_k 結果
            final_results = candidates[:top_k]
            
            # 格式化最終結果
            processed_results = []
            for i, result in enumerate(final_results):
                metadata = result.get('metadata', {})
                processed_results.append({
                    'rank': i + 1,
                    'content': result['content'],
                    'metadata': metadata,
                    'filename': metadata.get('filename', 'Unknown'),
                    'source': metadata.get('source', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', 0),
                    'total_chunks': metadata.get('total_chunks', 1),
                    'distance_score': round(result.get('distance_score', -1), 4),
                    'rerank_score': round(result.get('rerank_score', -1), 4),
                })
            
            logging.info(f"✅ 最終返回 {len(processed_results)} 個結果")
            return processed_results
            
        except Exception as e:
            logging.error(f"❌ 搜索過程發生錯誤: {e}")
            return []

    def get_all_files(self) -> List[str]:
        """
        獲取資料庫中所有文件的列表
        """
        try:
            all_results = self.vectordb.get(include=['metadatas'])
            metadatas = all_results.get('metadatas', [])
            
            filenames = set()
            for metadata in metadatas:
                if 'filename' in metadata:
                    filenames.add(metadata['filename'])
            
            return sorted(list(filenames))
            
        except Exception as e:
            logging.error(f"❌ 獲取文件列表時發生錯誤: {e}")
            return []
            
    def get_database_stats(self) -> Dict[str, Any]:
        """
        獲取資料庫統計信息
        """
        try:
            collection_count = self.vectordb._collection.count()
            all_files = self.get_all_files()
            
            stats = {
                'total_chunks': collection_count,
                'total_files': len(all_files),
                'files': all_files,
                'database_path': self.chroma_db_dir
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"❌ 獲取統計信息時發生錯誤: {e}")
            return {}

def print_search_results(results: List[Dict[str, Any]], max_content_length: int = 300):
    """
    格式化輸出搜索結果
    """
    if not results:
        print("❌ 沒有找到相關結果")
        return
    
    print(f"\n🎯 找到 {len(results)} 個相關結果:")
    print("=" * 80)
    
    for result in results:
        print(f"📍 排名: {result['rank']}")
        
        # 優先顯示 Rerank 分數
        if 'rerank_score' in result and result['rerank_score'] != -1:
            print(f"🎯 重排序分數: {result['rerank_score']:.4f}")
        
        # 顯示初始的距離分數
        if 'distance_score' in result and result['distance_score'] != -1:
            print(f"🔗 初始距離分數: {result['distance_score']:.4f}")

        print(f"📄 文件: {result['filename']}")
        print(f"📂 路徑: {result['source']}")
        
        if result['total_chunks'] > 1:
            print(f"📋 分塊: {result['chunk_id'] + 1}/{result['total_chunks']}")
        
        content = result['content']
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
            
        print(f"📝 內容預覽:")
        print(f"   {content}")
        print("-" * 80)

def interactive_query_mode(query_system: VectorQuerySystem):
    """
    互動式查詢模式
    """
    print("\n🚀 進入互動式查詢模式")
    print("輸入 'help' 查看可用命令")
    print("輸入 'exit' 或 'quit' 退出程式")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🔍 請輸入查詢 > ").strip()
            
            if not user_input:
                continue
            
            # 處理特殊命令
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("👋 再見！")
                break
            
            elif user_input.lower() == 'help':
                print("""
📖 可用命令:
 - 直接輸入文字進行語義搜索
 - 'files' - 顯示資料庫中所有文件
 - 'stats' - 顯示資料庫統計信息  
 - 'help' - 顯示此幫助信息
 - 'exit' 或 'quit' - 退出程式
 
🔧 搜索參數（可選）:
 - 在查詢後加上 ':數字' 指定最終結果數量，例如: "機器學習:5"
 - 在查詢後加上 ':數字:數字' 指定最終結果數量和初始候選集數量，例如: "機器學習:5:100"
 - 預設返回5個結果，候選集為50個
         """)
                continue
            
            elif user_input.lower() == 'files':
                files = query_system.get_all_files()
                print(f"\n📁 資料庫包含 {len(files)} 個文件:")
                for i, filename in enumerate(files, 1):
                    print(f"  {i}. {filename}")
                continue
            
            elif user_input.lower() == 'stats':
                stats = query_system.get_database_stats()
                print(f"\n📊 資料庫統計:")
                print(f"  總文檔片段: {stats.get('total_chunks', 0)}")
                print(f"  總文件數: {stats.get('total_files', 0)}")
                print(f"  資料庫路徑: {stats.get('database_path', 'Unknown')}")
                continue
            
            # 解析查詢參數
            top_k = 5
            initial_k = 50
            query_parts = user_input.split(':')
            user_query = query_parts[0].strip()
            if len(query_parts) >= 2 and query_parts[1].isdigit():
                top_k = int(query_parts[1])
            if len(query_parts) == 3 and query_parts[2].isdigit():
                initial_k = int(query_parts[2])

            # 執行搜索
            start_time = datetime.now()
            results = query_system.search(user_query, top_k=top_k, initial_k=initial_k)
            end_time = datetime.now()
            
            # 顯示結果
            print_search_results(results)
            print(f"\n⏱️ 搜索耗時: {(end_time - start_time).total_seconds():.3f} 秒")
            
        except KeyboardInterrupt:
            print("\n\n👋 程式已中斷，再見！")
            break
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")
            logging.error(f"互動模式錯誤: {e}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='向量資料庫查詢系統 (整合 Reranker)')
    parser.add_argument('--chroma_db_dir', type=str, default=r"D:\ArtificialIntelligenceCustomerService\vector_dbs\dev_dbs\tea_preview_list\cs_2048_co_10_min_1500",
                        help='向量資料庫目錄，預設為 "../vector_dbs/dev_dbs/"')
    parser.add_argument('--query', type=str, default=None,
                        help='直接執行查詢而不進入互動模式')
    parser.add_argument('--top_k', type=int, default=5,
                        help='最終返回結果數量，預設為5')
    parser.add_argument('--initial_k', type=int, default=50,
                        help='初始向量搜索的候選集數量，預設為50')
    parser.add_argument('--embedding_model', type=str, 
                        default="BAAI/bge-m3",
                        help='嵌入模型路徑')
    parser.add_argument('--reranker_model', type=str, 
                        default="BAAI/bge-reranker-v2-m3",
                        help='Reranker 模型路徑')
    
    args = parser.parse_args()
    
    try:
        print("🚀 初始化向量查詢系統...")
        query_system = VectorQuerySystem(
            chroma_db_dir=args.chroma_db_dir,
            embedding_model_name=args.embedding_model,
            reranker_model_name=args.reranker_model
        )
        
        stats = query_system.get_database_stats()
        print(f"\n📊 資料庫載入成功!")
        print(f"  📁 文檔片段數: {stats.get('total_chunks', 0)}")
        print(f"  📄 文件總數: {stats.get('total_files', 0)}")
        
        if args.query:
            print(f"\n🔍 執行查詢: '{args.query}'")
            results = query_system.search(args.query, top_k=args.top_k, initial_k=args.initial_k)
            print_search_results(results)
        else:
            interactive_query_mode(query_system)
            
    except FileNotFoundError as e:
        print(f"❌ 錯誤: {e}")
        print("請確保已經執行文檔向量化程式建立資料庫")
    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        logging.error(f"系統初始化失敗: {e}")

if __name__ == "__main__":
    main()