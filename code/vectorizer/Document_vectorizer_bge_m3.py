import os
import traceback
import logging
import argparse
from typing import Optional
import gc

# 設定PyTorch記憶體優化環境變數
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils.execution_logger import ExecutionLogger
from utils.vision_processor import VisionProcessor
from utils.document_reader import DocumentReader
from utils.text_splitter import TextSplitter
from utils.summarizer import Summarizer
from utils.vectorDB_manager import VectorDBManager
from FileFormatAnalyzer import FileFormatAnalyzer

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_file(file_path: str, document_reader: DocumentReader, text_splitter: TextSplitter,
                 vectordb_manager: VectorDBManager, execution_logger: ExecutionLogger, summarizer: Optional[Summarizer] = None):
    """
    處理單個文件：讀取、分割、摘要(可選)、加入向量資料庫。
    """
    try:
        logging.info(f"📄 處理檔案：{file_path}")
        raw_text = document_reader.load_document(file_path)
        
        if raw_text:
            print("="*40)
            print(f"成功從 {file_path} 讀取到的內容：")
            print(raw_text[:500] + "...") # 只列印前500個字元以避免內容過長
            print("="*40)

        if not raw_text:
            error_msg = "無法讀取文件或文件內容為空"
            execution_logger.log_error(error_msg, file_path)
            logging.error(f"❌ 無法讀取文件或文件內容為空：{file_path}")
            return False
            
        logging.info(f"📄 文件載入成功，內容長度：{len(raw_text)} 字元")
            
        logging.info(f"🔄 開始文本分割處理...")
        documents = text_splitter.split_text(raw_text, file_path)
        
        if summarizer:
            logging.info("📝 開始對每個文本塊生成摘要...")
            documents_with_summary = []
            for doc in documents:
                summary = summarizer.summarize_chunk(doc.page_content)
                doc.metadata["summary"] = summary
                documents_with_summary.append(doc)
                logging.info(f"已生成摘要，摘要長度：{len(summary)} 字元")
            documents_to_add = documents_with_summary
        else:
            documents_to_add = documents

        logging.info(f"🔄 開始向量化並加入資料庫...")
        vectordb_manager.add_documents(documents_to_add)
        
        logging.info(f"✅ 文件處理完成：{file_path}")
        return True

    except Exception as e:
        error_msg = f"處理文件時發生錯誤：{str(e)}"
        execution_logger.log_error(error_msg, file_path)
        logging.error(f"⚠️ 處理 {file_path} 時發生錯誤：{e}", exc_info=True)
        traceback.print_exc()
        return False


def process_all_files(source_dir, chroma_db_dir, summarize_enabled: bool):
    """
    處理所有文件並建立向量資料庫
    """
    execution_logger = ExecutionLogger("execution_log.txt")
    
    if not os.path.exists(source_dir):
        error_msg = f"源文件目錄不存在: {source_dir}"
        execution_logger.log_error(error_msg)
        logging.error(error_msg)
        return
    
    if not os.path.exists(chroma_db_dir):
        os.makedirs(chroma_db_dir)
        logging.info(f"創建了向量資料庫目錄: {chroma_db_dir}")
    
    input_files = []
    supported_extensions = ['.pdf', '.html', '.htm', '.txt', '.docx', '.json', '.csv', '.doc', '.odt']
    #supported_extensions = ['.pdf', '.html', '.htm', '.txt', '.docx', '.jpg', '.jpeg', '.png', '.json', '.csv', '.doc', '.odt']
    skipped_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.startswith('.'):
                continue
            file_ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(root, file)
            
            if file_ext in supported_extensions:
                input_files.append(file_path)
            else:
                skipped_files.append(file_path)
                execution_logger.log_skip(f"跳過不支援的文件格式: {file_ext}", file_path)
    
    if not input_files:
        error_msg = f"在 {source_dir} 中未找到任何支援的文件"
        execution_logger.log_error(error_msg)
        logging.error(error_msg)
        return
    
    logging.info(f"找到 {len(input_files)} 個待處理文件")
    logging.info(f"跳過 {len(skipped_files)} 個不支援的文件")
    logging.info(f"支援的格式: {', '.join(supported_extensions)}")
    
    try:
        vision_processor = VisionProcessor()
        document_reader = DocumentReader(vision_processor=vision_processor, execution_logger=execution_logger)
        text_splitter = TextSplitter(chunk_size=2048, chunk_overlap=50, min_chunk_size=400)
        vectordb_manager = VectorDBManager(chroma_db_dir)
        
        if summarize_enabled:
            summarizer = Summarizer()
        else:
            summarizer = None
            
    except Exception as e:
        error_msg = f"初始化組件時發生錯誤: {str(e)}"
        execution_logger.log_error(error_msg)
        logging.error(error_msg)
        return
    
    success_count = 0
    fail_count = 0
    skip_count = len(skipped_files)
     
    for i, file_path in enumerate(input_files, 1):
        logging.info(f"處理進度: {i}/{len(input_files)}")
        
        success = process_file(
            file_path=file_path,
            document_reader=document_reader,
            text_splitter=text_splitter,
            vectordb_manager=vectordb_manager,
            execution_logger=execution_logger,
            summarizer=summarizer
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            
        torch.cuda.empty_cache()
        gc.collect()
    
    execution_logger.finalize_log(success_count, fail_count, skip_count)
    
    logging.info(f"🎉 處理完成！")
    logging.info(f"✅ 成功處理: {success_count} 個文件")
    logging.info(f"❌ 處理失敗: {fail_count} 個文件")
    logging.info(f"⏭️ 跳過文件: {skip_count} 個文件")
    logging.info(f"📊 向量資料庫位置: {chroma_db_dir}")
    logging.info(f"📋 詳細執行日誌: execution_log.txt")


def main():
    """主函數"""
    print(f"PyTorch版本: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA可用。發現 {torch.cuda.device_count()} 個GPU設備")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"GPU {i}: {props.name}, 記憶體: {props.total_memory / 1024**3:.2f} GB")
            logging.info(f"目前使用的記憶體: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            logging.info(f"目前保留的記憶體: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        logging.warning("CUDA不可用，將使用CPU運行")
    
    torch.cuda.empty_cache()
    gc.collect()

    parser = argparse.ArgumentParser(description='文件向量化系統 - 將原始文件內容向量化並存入資料庫')
    parser.add_argument('--source_dir', type=str, default=r"C:\Users\IFangLab\Desktop\data_conclusion\教務處", 
                   help='源文件目錄，默認為 "../data"')
    # parser.add_argument('--source_dir', type=str, default="../../data", 
    #               help='源文件目錄，默認為 "../data"')
    parser.add_argument('--chroma_db_dir', type=str, default="../../tmp_db",
                      help='向量資料庫存儲目錄，默認為 "../tmp_db"')
    parser.add_argument('--summarize', action='store_true',
                      help='啟用對每個chunk進行摘要的功能')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_dir):
        logging.error(f"源文件目錄不存在: {args.source_dir}")
        return
    
    file_analyzer = FileFormatAnalyzer(args.source_dir)
    file_analyzer.analyze_folder()
    file_analyzer.generate_report()
    
    process_all_files(
        source_dir=args.source_dir,
        chroma_db_dir=args.chroma_db_dir,
        summarize_enabled=args.summarize
    )

if __name__ == "__main__":
    main()