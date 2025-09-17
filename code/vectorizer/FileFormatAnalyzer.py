import os
from collections import defaultdict
from pathlib import Path

class FileFormatAnalyzer:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.format_stats = defaultdict(int)
        
    def analyze_folder(self):
        """分析整個資料夾，統計檔案格式"""
        print(f"🔍 開始分析資料夾: {self.folder_path}")
        print("="*60)
        
        folder_path_obj = Path(self.folder_path)
        if not folder_path_obj.exists():
            print(f"❌ 資料夾不存在: {self.folder_path}")
            return
            
        all_files = list(folder_path_obj.rglob('*'))
        file_paths = [f for f in all_files if f.is_file()]
        
        total_files = len(file_paths)
        print(f"📁 找到 {total_files} 個檔案")
        print()
        
        for file_path in file_paths:
            file_extension = file_path.suffix.lower()
            self.format_stats[file_extension] += 1
            
        print(f"\n✅ 處理完成！總共處理了 {total_files} 個檔案")
    
    def generate_report(self):
        """生成檔案格式報告"""
        if not self.format_stats:
            print("❌ 沒有可用的統計資料")
            return
            
        print("\n" + "="*60)
        print("📊 檔案格式分佈報告")
        print("="*60)
        
        total_files = sum(self.format_stats.values())
        sorted_formats = sorted(self.format_stats.items(), key=lambda item: item[1], reverse=True)
        
        for ext, count in sorted_formats:
            percentage = count / total_files * 100
            print(f"   {ext or '無副檔名'}: {count} 個 ({percentage:.1f}%)")
            
        print("="*60)

# 使用範例
if __name__ == "__main__":
    folder_path = "./corpora"
    analyzer = FileFormatAnalyzer(folder_path)
    analyzer.analyze_folder()
    analyzer.generate_report()