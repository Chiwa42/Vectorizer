import os
import logging

# 設定PyTorch記憶體優化環境變數
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Hugging Face和語言模型相關
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Summarizer:
    """
    負責使用大型語言模型對文本塊進行摘要。
    """
    def __init__(self, model_id="google/gemma-3-12b-it", device="cuda:0"):
        self.device = device
        logging.info(f"正在載入摘要模型: {model_id}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map=device
            ).eval()
            logging.info("摘要模型載入成功。")
        except Exception as e:
            logging.error(f"載入摘要模型時發生錯誤: {e}")
            self.model = None

    def summarize_chunk(self, text: str) -> str:
        """
        對單個文本塊進行摘要。
        """
        if not self.model:
            logging.warning("摘要模型未載入，跳過摘要步驟。")
            return ""
        
        try:
            prompt = f"Generate a summary for the following text. The summary MUST be written in Traditional Chinese (Taiwan).\n\nText: {text}\n\nSummary:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = summary.replace(prompt, "").strip()
            
            print(f"\nGenerated summarizetion:{summary}\n")
            return summary
        except Exception as e:
            logging.error(f"生成摘要時發生錯誤: {e}")
            return ""