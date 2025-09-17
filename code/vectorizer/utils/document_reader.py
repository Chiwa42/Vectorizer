import os
import logging
import chardet
import json
import tempfile
import csv

# 文件處理相關
import html2text
import bs4
import re
import pandas as pd
import win32com.client as win32
import tempfile
import docx
import docx2txt
import win32com.client as win32
import pydocx
from odf.opendocument import load
from odf.text import P
from odf import teletype

# PDF轉圖片相關
import fitz
import io
from PIL import Image

class DocumentReader:
    """
    負責讀取不同格式的文件。
    """
    def __init__(self, vision_processor=None, execution_logger=None):
        self.h = html2text.HTML2Text()
        self.h.ignore_links = False
        self.vision_processor = vision_processor
        self.execution_logger = execution_logger

    def load_document(self, file_path: str) -> str:
        """
        讀取文件並返回文本內容。
        """
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.pdf':
                # 新增邏輯: 首先嘗試純文字提取
                logging.info(f"PDF文件開始純文字提取：{file_path}")
                text = self.process_pdf_as_text(file_path)

                if not text:
                    # 如果純文字提取失敗，退回使用視覺化處理
                    if self.vision_processor is None:
                        error_msg = "需要提供 VisionProcessor 實例才能處理PDF文件"
                        if self.execution_logger:
                            self.execution_logger.log_error(error_msg, file_path)
                        raise ValueError(error_msg)
                    
                    logging.info(f"純文字提取失敗，改為PDF文件轉換為圖片處理：{file_path}")
                    text = self.process_pdf_as_images(file_path)

            elif file_ext in ['.jpg', '.jpeg', '.png']:
                if self.vision_processor is None:
                    error_msg = "需要提供 VisionProcessor 實例才能處理視覺化文件"
                    if self.execution_logger:
                        self.execution_logger.log_error(error_msg, file_path)
                    raise ValueError(error_msg)
                
                logging.info(f"使用視覺化功能處理圖片：{file_path}")
                text = self.vision_processor.extract_text_with_vision(file_path)

            elif file_ext == '.html' or file_ext == '.htm':
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                soup = bs4.BeautifulSoup(html_content, 'html.parser')
                
                for script in soup(["script", "style"]):
                    script.extract()
                
                text = soup.get_text(separator=' ', strip=True)
                
                text = re.sub(r'\s+', ' ', text).strip()
            
                if not text:
                    text = self.h.handle(html_content)
                    
                logging.info(f"HTML 解析完成，內容長度：{len(text)} 字元")
                
            elif file_ext == '.txt':
                text = self.read_text_file(file_path)
                
            elif file_ext == '.doc':
                try:
                    text = pydocx.PyDocX().parse_docx_file(file_path)
                    text = html2text.html2text(text)
                    logging.info("使用 pydocx 成功讀取 .doc 文件。")
                except ImportError:
                    self.execution_logger.log_warning("未安裝 pydocx，將嘗試其他方法讀取 .doc 文件。請安裝: pip install pydocx")
                    text = ""
                except Exception as e:
                    self.execution_logger.log_error(f"使用 pydocx 讀取文件時發生錯誤：{e}", file_path)
                    text = ""
                    
                if not text:
                    try:

                        word = win32.Dispatch("Word.Application")
                        word.Visible = False
                        doc = word.Documents.Open(file_path)
                        text = doc.Content.Text
                        doc.Close()
                        word.Quit()
                        logging.info("使用 win32com 成功讀取 .doc 文件。")
                    except Exception:
                        try:

                            text = docx2txt.process(file_path)
                            logging.warning("使用 docx2txt 讀取 .doc 文件，這可能不是最準確的方式。")
                        except ImportError:
                            error_msg = "處理 .doc 文件需要安裝 pydocx 或 docx2txt，或在 Windows 環境下使用 Microsoft Word"
                            if self.execution_logger:
                                self.execution_logger.log_error(error_msg, file_path)
                            raise ValueError(error_msg)
            
            elif file_ext == '.docx':
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])

            elif file_ext == '.odt':
                text = self.read_odt_file(file_path)

            elif file_ext == '.json':
                text = self.read_json_file(file_path)

            elif file_ext == '.csv':
                text = self.read_csv_file(file_path)

            else:
                error_msg = f"不支援的文件格式：{file_ext}"
                if self.execution_logger:
                    self.execution_logger.log_error(error_msg, file_path)
                raise ValueError(error_msg)
            
            if not text.strip():
                warning_msg = f"文件內容為空"
                if self.execution_logger:
                    self.execution_logger.log_warning(warning_msg, file_path)
                logging.warning(f"警告：文件 {file_path} 內容為空")
                
            return text
        except Exception as e:
            error_msg = f"讀取文件時發生錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(f"讀取文件 {file_path} 時發生錯誤：{e}", exc_info=True)
            return ""

    def process_pdf_as_text(self, pdf_path: str) -> str:
        """
        嘗試直接從PDF提取純文字內容
        """
        pdf_document = None
        try:
            pdf_document = fitz.open(pdf_path)
            extracted_texts = [page.get_text("text") for page in pdf_document]
            full_text = "\n".join(extracted_texts)
            
            if full_text.strip():
                logging.info(f"PDF純文字提取完成，共 {len(pdf_document)} 頁，總文字長度：{len(full_text)} 字元")
            else:
                logging.warning(f"PDF純文字提取失敗，文件 {pdf_path} 可能為掃描檔。")
            
            return full_text
        except Exception as e:
            logging.error(f"純文字提取PDF時發生錯誤: {e}", exc_info=True)
            return ""
        finally:
            if pdf_document:
                pdf_document.close()

    def read_json_file(self, file_path: str) -> str:
        """
        讀取JSON文件並轉換為文本格式
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            text = self.json_to_text(json_data, file_path)
            
            logging.info(f"JSON文件解析完成，內容長度：{len(text)} 字元")
            return text
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON格式錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(f"JSON文件 {file_path} 格式錯誤：{e}")
            return ""
        except UnicodeDecodeError as e:
            try:
                with open(file_path, 'r', encoding='big5') as f:
                    json_data = json.load(f)
                text = self.json_to_text(json_data, file_path)
                logging.info(f"使用Big5編碼成功讀取JSON文件：{file_path}")
                return text
            except Exception as e2:
                error_msg = f"編碼錯誤，嘗試多種編碼均失敗：{str(e2)}"
                if self.execution_logger:
                    self.execution_logger.log_error(error_msg, file_path)
                logging.error(f"JSON文件 {file_path} 編碼錯誤：{e2}")
                return ""
        except Exception as e:
            error_msg = f"讀取JSON文件時發生未知錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(f"讀取JSON文件 {file_path} 時發生錯誤：{e}")
            return ""

    def json_to_text(self, json_data, file_path: str) -> str:
        """
        將JSON數據轉換為可讀的文本格式
        """
        try:
            filename = os.path.basename(file_path)
            text_parts = [f"文件名稱：{filename}\n"]
            
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
            error_msg = f"JSON轉文本時發生錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(f"JSON轉文本處理 {file_path} 時發生錯誤：{e}")
            return json.dumps(json_data, ensure_ascii=False, indent=2)

    def process_pdf_as_images(self, pdf_path: str) -> str:
        """
        將PDF轉換為圖片，然後使用視覺化功能提取文字
        """
        pdf_document = None
        temp_image_path = None
        try:
            pdf_document = fitz.open(pdf_path)
            extracted_texts = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                    temp_image_path = temp_file.name
                    image.save(temp_image_path)

                try:
                    page_text = self.vision_processor.extract_text_with_vision(temp_image_path)
                    
                    print("="*40)
                    print(f"從 PDF 第 {page_num + 1} 頁提取到的文字：")
                    print(page_text)
                    print("="*40)

                    if page_text.strip():
                        extracted_texts.append(f"=== 第 {page_num + 1} 頁 ===\n{page_text}")
                    else:
                        warning_msg = f"PDF第 {page_num + 1} 頁未提取到文字內容"
                        if self.execution_logger:
                            self.execution_logger.log_warning(warning_msg, pdf_path)
                        logging.warning(warning_msg)
                        
                finally:
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
                
            full_text = "\n\n".join(extracted_texts)
            
            if full_text:
                logging.info(f"PDF轉圖片處理完成，共 {len(pdf_document)} 頁，總文字長度：{len(full_text)} 字元")
            else:
                warning_msg = f"PDF文件未能提取到任何文字內容"
                if self.execution_logger:
                    self.execution_logger.log_warning(warning_msg, pdf_path)
                logging.warning(f"PDF文件 {pdf_path} 未能提取到任何文字內容")
                
            return full_text
        
        except Exception as e:
            error_msg = f"處理PDF文件時發生錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, pdf_path)
            logging.error(f"處理PDF文件 {pdf_path} 時發生錯誤：{e}", exc_info=True)
            return ""
        finally:
            if pdf_document:
                pdf_document.close()
            
    def read_text_file(self, file_path):
        """讀取文本文件內容，自動檢測編碼"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                detected_encoding = result['encoding']
                confidence = result['confidence']

            if detected_encoding and confidence > 0.5:
                try:
                    with open(file_path, 'r', encoding=detected_encoding) as f:
                        content = f.read()
                        logging.info(f"使用檢測到的編碼 {detected_encoding} (信心度: {confidence:.2f}) 讀取文件: {os.path.basename(file_path)}")
                        return content
                except UnicodeDecodeError:
                    pass
            
            encodings = ['utf-8', 'big5', 'gbk', 'cp950', 'cp936', 'latin-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        logging.info(f"使用 {encoding} 編碼成功讀取文件: {os.path.basename(file_path)}")
                        return content
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    error_msg = f"讀取文件時發生錯誤: {str(e)}"
                    if self.execution_logger:
                        self.execution_logger.log_error(error_msg, file_path)
                    logging.error(f"讀取文件時發生錯誤: {e}")
                    return ""
            
            with open(file_path, 'rb') as f:
                content = f.read().decode('latin-1')
                warning_msg = f"使用 latin-1 (失敗安全模式) 讀取文件，可能有亂碼"
                if self.execution_logger:
                    self.execution_logger.log_warning(warning_msg, file_path)
                logging.warning(f"使用 latin-1 (失敗安全模式) 讀取文件: {os.path.basename(file_path)}，可能有亂碼")
                return content
                
        except Exception as e:
            error_msg = f"讀取文件時發生錯誤: {str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(f"讀取文件 {file_path} 時發生錯誤：{e}")
            return ""
        
    def read_csv_file(self, file_path: str) -> str:
        try:
            filename = os.path.basename(file_path)
            
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='big5')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='gbk')
            
            structured_content = self.csv_to_structured_text(df, filename)
            
            logging.info(f"CSV文件讀取完成，共 {len(df)} 行 {len(df.columns)} 列，內容長度：{len(structured_content)} 字元")
            return structured_content
            
        except Exception as e:
            try:
                return self.read_csv_with_standard_library(file_path)
            except Exception as e2:
                error_msg = f"讀取CSV文件時發生錯誤：{str(e)}，備用方法也失敗：{str(e2)}"
                if self.execution_logger:
                    self.execution_logger.log_error(error_msg, file_path)
                logging.error(f"讀取CSV文件 {file_path} 時發生錯誤：{e}")
                return ""

    def read_csv_with_standard_library(self, file_path: str) -> str:
        text_parts = []
        try:
            encodings = ['utf-8', 'big5', 'gbk', 'cp950']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', newline='', encoding=encoding) as f:
                        csv_reader = csv.reader(f)
                        header = next(csv_reader)
                        text_parts.append("CSV 文件內容：")
                        text_parts.append(", ".join(header))
                        for row in csv_reader:
                            text_parts.append(", ".join(row))
                        return "\n".join(text_parts)
                except (UnicodeDecodeError, StopIteration):
                    continue
            return "無法以任何支援的編碼讀取CSV文件。"
        except Exception as e:
            logging.error(f"使用標準csv模組讀取文件時發生錯誤：{e}")
            raise e
            
    def csv_to_structured_text(self, df: pd.DataFrame, filename: str) -> str:
        try:
            text_parts = [f"CSV文件：{filename}"]
            text_parts.append(f"數據概要：共 {len(df)} 行，{len(df.columns)} 列\n")
            
            columns = df.columns.tolist()
            text_parts.append("欄位名稱：" + "、".join(columns))
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
                text_parts.append(f"\n註：僅顯示前 {max_rows_to_process} 行，總共有 {len(df)} 行資料")
            
            return "\n".join(text_parts)
        except Exception as e:
            error_msg = f"轉換CSV為文本時發生錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, filename)
            logging.error(f"轉換CSV為文本時發生錯誤：{e}")
            return f"CSV文件：{filename}\n{df.to_string()}"
        
    def read_odt_file(self, file_path: str) -> str:
        """
        使用 odfpy 讀取 .odt 文件並提取文本內容
        """
        try:
            doc = load(file_path)
            text_parts = []
            
            # 使用 odfpy 內建的 teletype.extractText() 函數
            # 這是最可靠的方法，它會處理所有的子節點並提取純文字
            full_text = teletype.extractText(doc.text)
            text_parts.append(full_text)
            
            text = "\n".join(text_parts)
            logging.info(f"使用 odfpy 成功讀取 .odt 文件，內容長度：{len(text)} 字元")
            return text
        except ImportError:
            error_msg = "處理 .odt 文件需要安裝 odfpy，請執行: pip install odfpy"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"讀取 .odt 文件時發生錯誤：{str(e)}"
            if self.execution_logger:
                self.execution_logger.log_error(error_msg, file_path)
            logging.error(f"讀取文件 {file_path} 時發生錯誤：{e}", exc_info=True)
            return ""