import logging
from datetime import datetime

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ExecutionLogger:
    """
    執行日誌記錄器，專門記錄錯誤和警告信息
    """
    
    def __init__(self, log_file="execution_log.txt"):
        self.log_file = log_file
        self.errors = []
        self.warnings = []
        self.start_time = datetime.now()
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"文檔向量化執行日誌 - 開始時間: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
    
    def log_error(self, message: str, file_path: str = None):
        """記錄錯誤信息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if file_path:
            error_msg = f"[{timestamp}] ERROR - {file_path}: {message}"
        else:
            error_msg = f"[{timestamp}] ERROR - {message}"
        
        self.errors.append(error_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + "\n")
        
        logging.error(error_msg)
    
    def log_warning(self, message: str, file_path: str = None):
        """記錄警告信息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if file_path:
            warning_msg = f"[{timestamp}] WARNING - {file_path}: {message}"
        else:
            warning_msg = f"[{timestamp}] WARNING - {message}"
        
        self.warnings.append(warning_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(warning_msg + "\n")
    
        logging.warning(warning_msg)
    
    def log_skip(self, message: str, file_path: str = None):
        """記錄跳過文件的信息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if file_path:
            skip_msg = f"[{timestamp}] SKIP - {file_path}: {message}"
        else:
            skip_msg = f"[{timestamp}] SKIP - {message}"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(skip_msg + "\n")
        
        logging.info(skip_msg)

    def write_to_log(self, content: str):
        """將任意內容寫入日誌文件"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
    
    def finalize_log(self, success_count: int, fail_count: int, skip_count: int):
        """完成日誌記錄"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = f"""
處理完成時間: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
總執行時間: {duration}
成功處理: {success_count} 個文件
處理失敗: {fail_count} 個文件
跳過文件: {skip_count} 個文件
錯誤總數: {len(self.errors)} 個
警告總數: {len(self.warnings)} 個

{"=" * 60}
"""
        self.write_to_log(summary)
        
        logging.info(f"執行日誌已保存至: {self.log_file}")