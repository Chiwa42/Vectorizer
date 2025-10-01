import os
import traceback
import logging
import argparse
from typing import Optional
import gc
from collections import defaultdict
import glob

# 設定PyTorch記憶體優化環境變數
import torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 假設這些工具類別都已定義
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
                 vectordb_manager: VectorDBManager, execution_logger: ExecutionLogger, summarizer: Optional[Summarizer] = None) -> bool:
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


def process_subdirectories(base_source_dir: str, base_chroma_db_dir: str, summarize_enabled: bool, single_db_mode: bool):
    """
    處理所有子目錄並根據 single_db_mode 建立向量資料庫
    """
    # 決定最終的向量資料庫目錄
    if single_db_mode:
        # 單一模式: 所有子目錄文件都匯入到一個資料庫
        target_db_dir = base_chroma_db_dir
        logging.info(f"**啟用單一資料庫模式**：所有文件將匯入到 {target_db_dir}")
        
        # 尋找所有子目錄中的文件
        all_subdirs = [os.path.join(base_source_dir, d) for d in os.listdir(base_source_dir) if os.path.isdir(os.path.join(base_source_dir, d))]
        if not all_subdirs:
            logging.error(f"在 {base_source_dir} 中未找到任何子目錄。")
            return
            
        # 為了保持與 process_single_directory 的邏輯一致，我們將所有子目錄視為一個虛擬的「單一目錄」
        # 但實際遍歷時需要遞歸尋找所有文件
        process_single_directory(base_source_dir, target_db_dir, summarize_enabled, is_recursive=True)

    else:
        # 分散模式: 每個子目錄建立一個向量資料庫
        subdirectories = [d for d in os.listdir(base_source_dir) if os.path.isdir(os.path.join(base_source_dir, d))]

        if not subdirectories:
            logging.error(f"在 {base_source_dir} 中未找到任何子目錄。請確認此路徑下包含各系所的資料夾。")
            return

        for subdir_name in subdirectories:
            source_dir = os.path.join(base_source_dir, subdir_name)
            chroma_db_dir = os.path.join(base_chroma_db_dir, subdir_name)

            logging.info(f"\n======== 開始處理子目錄：{source_dir} ========")

            process_single_directory(source_dir, chroma_db_dir, summarize_enabled, is_recursive=False) # 這裡只處理單一層級

            logging.info(f"======== 子目錄 {subdir_name} 處理完成 ========")

def process_single_directory(source_dir: str, chroma_db_dir: str, summarize_enabled: bool, is_recursive: bool = False):
    """
    處理單一目錄（或遞歸處理基底目錄下的所有文件）並建立向量資料庫
    :param is_recursive: 是否遞歸遍歷 source_dir 下的所有文件
    """
    if not os.path.exists(chroma_db_dir):
        os.makedirs(chroma_db_dir)
        logging.info(f"創建了向量資料庫目錄: {chroma_db_dir}")

    execution_log_path = os.path.join(chroma_db_dir, "execution_log.txt")
    execution_logger = ExecutionLogger(execution_log_path)
    
    if not os.path.exists(source_dir):
        error_msg = f"源文件目錄不存在: {source_dir}"
        execution_logger.log_error(error_msg)
        logging.error(error_msg)
        return
    
    input_files = []
    supported_extensions = ['.pdf', '.html', '.htm', '.txt', '.docx','.json', '.csv', '.doc', '.odt']
    
    # 新增一個字典來追蹤文件格式的統計
    file_stats = {'valid': defaultdict(int), 'invalid': defaultdict(int)}
    
    # 根據 is_recursive 決定遍歷方式
    if is_recursive:
        # 遞歸查找所有文件
        search_path = os.path.join(source_dir, "**/*")
        file_list = glob.glob(search_path, recursive=True)
    else:
        # 只查找當前目錄下的文件 (process_subdirectories 中會對每個子目錄調用)
        file_list = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    for file_path in file_list:
        if os.path.basename(file_path).startswith('.'):
            continue
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in supported_extensions:
            input_files.append(file_path)
        else:
            execution_logger.log_skip(f"跳過不支援的文件格式: {file_ext}", file_path)
    
    if not input_files:
        error_msg = f"在 {source_dir} 中未找到任何支援的文件"
        execution_logger.log_error(error_msg)
        logging.error(error_msg)
        return
    
    logging.info(f"找到 {len(input_files)} 個待處理文件")
    logging.info(f"支援的格式: {', '.join(supported_extensions)}")
    
    try:
        vision_processor = VisionProcessor()
        document_reader = DocumentReader(vision_processor=vision_processor, execution_logger=execution_logger)
        text_splitter = TextSplitter(chunk_size=2048, chunk_overlap=100, min_chunk_size=1500)
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
    skip_count = 0
    
    # 在處理迴圈前，先獲取所有已處理過的文件路徑
    try:
        processed_file_paths = vectordb_manager.get_processed_paths()
        logging.info(f"已從資料庫讀取 {len(processed_file_paths)} 個已處理文件路徑。")
    except Exception as e:
        logging.warning(f"無法從資料庫讀取已處理文件路徑，將從頭處理所有文件。錯誤: {e}")
        processed_file_paths = set()

    for i, file_path in enumerate(input_files, 1):
        logging.info(f"處理進度: {i}/{len(input_files)}")
        
        # 檢查文件是否已處理過
        if file_path in processed_file_paths:
            logging.info(f"⏭️ 文件已存在於資料庫中，跳過處理: {file_path}")
            execution_logger.log_skip(f"文件已存在於資料庫中，跳過處理", file_path)
            skip_count += 1
            continue

        # 這裡需要修改 process_file 函數，讓它能返回成功與否
        success = process_file(
            file_path=file_path,
            document_reader=document_reader,
            text_splitter=text_splitter,
            vectordb_manager=vectordb_manager,
            execution_logger=execution_logger,
            summarizer=summarizer
        )
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if success:
            success_count += 1
            file_stats['valid'][file_ext] += 1
        else:
            fail_count += 1
            file_stats['invalid'][file_ext] += 1
            
        torch.cuda.empty_cache()
        gc.collect()
    
    # 記錄文件格式統計
    execution_logger.write_to_log("\n--- 文件格式處理統計 ---")
    for ext in supported_extensions:
        valid_count = file_stats['valid'][ext]
        invalid_count = file_stats['invalid'][ext]
        execution_logger.write_to_log(f"格式 {ext}: 有效文件數={valid_count}, 無效文件數={invalid_count}")
    
    execution_logger.finalize_log(success_count, fail_count, skip_count)
    
    logging.info(f"🎉 處理完成！")
    logging.info(f"✅ 成功處理: {success_count} 個文件")
    logging.info(f"❌ 處理失敗: {fail_count} 個文件")
    logging.info(f"⏭️ 跳過文件: {skip_count} 個文件")
    logging.info(f"📊 向量資料庫位置: {chroma_db_dir}")
    logging.info(f"📋 詳細執行日誌: {execution_log_path}")


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
    parser.add_argument('--source_dir', type=str, default=r"C:\Users\IFangLab\Desktop\data_conclusion", 
                        help='包含各系所資料夾的基底源文件目錄')
    parser.add_argument('--chroma_db_dir', type=str, default="../../tmp_db",
                        help='向量資料庫的基底存儲目錄。若為分散模式，每個子目錄會在此處建立對應的DB；若為單一模式，所有文件將匯入到此目錄下的DB。')
    parser.add_argument('--summarize', action='store_true',
                        help='啟用對每個chunk進行摘要的功能')
    parser.add_argument('--single_db', action='store_true',
                        help='啟用單一資料庫模式。如果設定此旗標，所有子目錄中的文件將被匯入到一個單一的向量資料庫（--chroma_db_dir）。否則，每個子目錄將有自己的資料庫。')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_dir):
        logging.error(f"源文件基底目錄不存在: {args.source_dir}")
        return
    
    file_analyzer = FileFormatAnalyzer(args.source_dir)
    file_analyzer.analyze_folder()
    file_analyzer.generate_report()

    process_subdirectories(
        base_source_dir=args.source_dir,
        base_chroma_db_dir=args.chroma_db_dir,
        summarize_enabled=args.summarize,
        single_db_mode=args.single_db # 傳遞新參數
    )

if __name__ == "__main__":
    main()