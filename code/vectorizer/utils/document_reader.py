import os
import logging
import chardet
import json
import tempfile
import csv
import html2text
import bs4
import re
import pandas as pd
import win32com.client as win32
import docx
import docx2txt
import fitz
import pytesseract
from PIL import Image
import io
import traceback
from odf.opendocument import load
from odf.text import P
from odf import teletype
import weasyprint
from weasyprint import HTML
import pydocx
from pydocx import PyDocX
import subprocess

# 設定 Tesseract 路徑
# 如果 Tesseract 已加入系統 PATH，此行可註解掉
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 設定日誌格式
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentReader:
    """
    負責讀取不同格式的文件並返回文本內容。
    """
    def __init__(self, vision_processor=None, execution_logger=None):
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.vision_processor = vision_processor
        self.execution_logger = execution_logger

    def load_document(self, file_path: str) -> str:
        """
        讀取文件並返回文本內容，根據文件擴展名選擇適當的方法。
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                logging.info(f"PDF 文件先嘗試純文字提取: {file_path}")
                text = self.process_pdf_smartly(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png']:
                if self.vision_processor is None:
                    raise ValueError("需要提供 VisionProcessor 實例才能處理圖片文件。")
                
                logging.info(f"使用視覺化功能處理圖片: {file_path}")
                text = self.vision_processor.extract_text_with_vision(file_path)
            elif file_ext in ['.html', '.htm']:
                text = self.read_html_file(file_path)
            elif file_ext == '.txt':
                text = self.read_text_file(file_path)
            elif file_ext == '.doc':
                text = self.read_doc_file(file_path)
            elif file_ext == '.docx':
                # 新增的 .docx 處理方法
                text = self.read_docx_file(file_path)
            elif file_ext == '.odt':
                text = self.read_odt_file(file_path)
            elif file_ext == '.json':
                text = self.read_json_file(file_path)
            elif file_ext == '.csv':
                text = self.read_csv_file(file_path)
            else:
                raise ValueError(f"不支援的文件格式: {file_ext}")
            
            if not text.strip():
                logging.warning(f"文件 {file_path} 內容為空。")
            
            return text
        except Exception as e:
            logging.error(f"讀取文件 {file_path} 時發生錯誤: {e}", exc_info=True)
            if self.execution_logger:
                self.execution_logger.log_error(f"讀取文件時發生錯誤: {str(e)}", file_path)
            return ""

    def read_html_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            soup = bs4.BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                text = self.h.handle(html_content)
            logging.info(f"HTML 解析完成，內容長度: {len(text)} 字元")
            return text
        except Exception as e:
            logging.error(f"解析 HTML 文件 {file_path} 時發生錯誤: {e}", exc_info=True)
            return ""
        
    def read_doc_file(self, file_path: str) -> str:
        try:
            word = win32.Dispatch("Word.Application")
            word.Visible = False
            doc = word.Documents.Open(file_path)
            
            # 建立一個暫存的 .docx 檔名
            temp_docx_path = file_path + "x"
            
            # 將 .doc 存為 .docx
            doc.SaveAs(temp_docx_path, 16) # 16 for wdFormatXMLDocument (docx)
            doc.Close()
            word.Quit()
            
            # 使用 docx2txt 讀取這個暫存檔
            text = docx2txt.process(temp_docx_path)
            
            # 讀取完畢後，刪除暫存檔
            os.remove(temp_docx_path)
            
            # 回傳讀取到的內容
            return text
        except Exception as e:
            logging.error(f"處理 .doc 文件 {file_path} 時發生錯誤: {e}")
            return ""
        
    def read_docx_file(self, file_path: str) -> str:
        """
        使用 docx2txt 讀取 .docx 文件並返回文本內容。
        """
        try:
            logging.info(f"開始使用 docx2txt 讀取 .docx 文件: {file_path}")
            text = docx2txt.process(file_path)
            logging.info(f".docx 文件讀取完成，內容長度: {len(text)} 字元")
            return text
        except Exception as e:
            logging.error(f"讀取 .docx 文件 {file_path} 時發生錯誤: {e}", exc_info=True)
            return ""

    def read_odt_file(self, file_path: str) -> str:
        """
        嘗試使用 odfpy 讀取 .odt 文件。若失敗，則轉換為 PDF 再處理。
        """
        try:
            doc = load(file_path)
            odt_text = teletype.extractText(doc.text)
            
            if odt_text.strip():
                logging.info(f"使用 odfpy 成功讀取 .odt 文件：{file_path}")
                return odt_text
            else:
                raise Exception("ODT 文件純文字內容為空，嘗試備用方案。")
        
        except Exception as e:
            logging.warning(f"ODT 文件純文字讀取失敗，原因：{e}，正在嘗試轉換為 PDF 後處理。")
            
            temp_pdf_path = None
            try:
                # 使用 weasyprint 將 ODT 轉換為 PDF
                # 這裡需要一個中間格式，例如先轉為 HTML 再轉 PDF
                # 考慮到 weasyprint 不直接支援 ODT，我們需要一個中間步驟，例如Pandoc
                # 由於您要求不要用Pandoc，這裡需要一個替代方案，最簡單的是使用LibreOffice的CLI
                # 以下是使用 LibreOffice 的範例，需要確保 LibreOffice 已安裝
                output_dir = tempfile.gettempdir()
                temp_pdf_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + '.pdf')
                
                if os.name == 'nt': # Windows
                    libreoffice_cmd = 'soffice.exe'
                else: # Linux/macOS
                    libreoffice_cmd = 'libreoffice'
                
                cmd = [
                    libreoffice_cmd,
                    '--headless',
                    '--convert-to', 'pdf',
                    file_path,
                    '--outdir', output_dir
                ]
                
                subprocess.run(cmd, check=True, timeout=60)
                
                if os.path.exists(temp_pdf_path):
                    logging.info(f"成功將 ODT 轉換為 PDF: {temp_pdf_path}")
                    pdf_text = self.process_pdf_smartly(temp_pdf_path)
                    os.remove(temp_pdf_path)
                    return pdf_text
                else:
                    raise Exception("轉換為 PDF 失敗，未產生檔案。")
                
            except FileNotFoundError:
                logging.error("無法找到 LibreOffice，請確保已安裝並將其添加到系統 PATH。")
                return ""
            except subprocess.CalledProcessError as e2:
                logging.error(f"LibreOffice 轉換文件時發生錯誤：{e2.stderr}")
                return ""
            except Exception as e3:
                logging.error(f"轉換 ODT 文件時發生未知錯誤: {e3}", exc_info=True)
                return ""
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

    def read_json_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            text = self.json_to_text(json_data, file_path)
            logging.info(f"JSON文件解析完成，內容長度: {len(text)} 字元")
            return text
        except json.JSONDecodeError as e:
            logging.error(f"JSON文件 {file_path} 格式錯誤: {e}")
            return ""
        except UnicodeDecodeError as e:
            try:
                with open(file_path, 'r', encoding='big5') as f:
                    json_data = json.load(f)
                text = self.json_to_text(json_data, file_path)
                logging.info(f"使用Big5編碼成功讀取JSON文件: {file_path}")
                return text
            except Exception as e2:
                logging.error(f"JSON文件 {file_path} 編碼錯誤，嘗試多種編碼均失敗: {e2}")
                return ""
        except Exception as e:
            logging.error(f"讀取JSON文件 {file_path} 時發生未知錯誤: {e}")
            return ""

    def json_to_text(self, json_data, file_path: str) -> str:
        try:
            filename = os.path.basename(file_path)
            text_parts = [f"文件名稱: {filename}\n"]
            
            def process_json_object(obj, level=0):
                indent = "  " * level
                text_content = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, (dict, list)):
                            text_content.append(f"{indent}{key}:")
                            text_content.extend(process_json_object(value, level + 1))
                        else:
                            text_content.append(f"{indent}{key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            text_content.append(f"{indent}項目 {i + 1}:")
                            text_content.extend(process_json_object(item, level + 1))
                        else:
                            text_content.append(f"{indent}項目 {i + 1}: {item}")
                else:
                    text_content.append(f"{indent}{obj}")
                return text_content
            
            text_parts.extend(process_json_object(json_data))
            return "\n".join(text_parts)
        except Exception as e:
            logging.error(f"JSON轉文本時發生錯誤: {e}")
            return json.dumps(json_data, ensure_ascii=False, indent=2)

    def read_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            if detected_encoding and confidence > 0.5:
                try:
                    with open(file_path, 'r', encoding=detected_encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    pass
            encodings = ['utf-8', 'big5', 'gbk', 'cp950', 'cp936', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            with open(file_path, 'rb') as f:
                content = f.read().decode('latin-1')
                logging.warning(f"文件 {os.path.basename(file_path)} 使用 latin-1 (失敗安全模式) 讀取，可能有亂碼。")
                return content
        except Exception as e:
            logging.error(f"讀取文本文件 {file_path} 時發生錯誤: {e}")
            return ""

    def read_csv_file(self, file_path: str) -> str:
        try:
            df = None
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='big5')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='gbk')
            
            if df is not None:
                return self.csv_to_structured_text(df, os.path.basename(file_path))
            else:
                raise Exception("無法讀取 CSV 文件")
        except Exception as e:
            logging.error(f"讀取CSV文件 {file_path} 時發生錯誤: {e}")
            return ""

    def csv_to_structured_text(self, df: pd.DataFrame, filename: str) -> str:
        text_parts = [f"CSV 文件: {filename}"]
        text_parts.append(f"數據概要: 共 {len(df)} 行，{len(df.columns)} 列\n")
        columns = df.columns.tolist()
        text_parts.append("欄位名稱: " + "、".join(columns))
        text_parts.append("")
        max_rows_to_process = min(len(df), 1000)
        for index, row in df.head(max_rows_to_process).iterrows():
            row_parts = []
            for col in columns:
                value = row[col]
                if pd.isna(value):
                    continue
                str_value = str(value).strip()
                if str_value and str_value.lower() not in ['nan', 'null', '']:
                    row_parts.append(f"{col}: {str_value}")
            if row_parts:
                text_parts.append(f"第 {index + 1} 行資料 - " + "；".join(row_parts))
        if len(df) > max_rows_to_process:
            text_parts.append(f"\n註: 僅顯示前 {max_rows_to_process} 行，總共有 {len(df)} 行資料")
        return "\n".join(text_parts)

    def is_garbage_text(self, text, min_valid_ratio=0.2, min_text_len=50):
        """
        更精確地判斷文字是否為亂碼，考慮中、英、數。
        """
        if not text or len(text) < min_text_len:
            return True
        
        total_chars = len(text)
        # 統計中文字元和英數字元的總數
        valid_chars = len(re.findall(r'[\u4e00-\u9fff\w]', text))
        
        valid_ratio = valid_chars / total_chars
        
        # 如果有效字元（中文字、英文字、數字）的比例過低，則判定為亂碼。
        return valid_ratio < min_valid_ratio

    def process_pdf_smartly(self, file_path: str) -> str:
        """
        混合式提取 PDF 文字：先純文字，失敗則轉 OCR。
        """
        full_text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # 檢查提取結果是否為亂碼或太短
                if self.is_garbage_text(text):
                    logging.info(f"第 {page_num + 1} 頁純文字提取失敗或為亂碼，正在切換為 OCR 模式...")
                    
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    ocr_text = pytesseract.image_to_string(img, lang='chi_tra')
                    full_text += ocr_text + "\n"
                else:
                    logging.info(f"第 {page_num + 1} 頁成功純文字提取。")
                    full_text += text + "\n"
            
            doc.close()
            return full_text
        except Exception as e:
            logging.error(f"處理 PDF 文件時發生錯誤: {e}", exc_info=True)
            return None