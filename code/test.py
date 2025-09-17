import time
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch
import os

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# 設定下載速度上限（例如：每秒 1 MB）
download_speed_limit = 1 * 1024 * 1024  # 單位：位元組/秒

# 定義下載路徑
local_dir = f"./{model_id.replace('/', '_')}"
os.makedirs(local_dir, exist_ok=True)

def download_file_with_limit(url, local_path, speed_limit):
    """
    下載檔案並限制速度
    """
    print(f"開始下載 {url} 到 {local_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果請求失敗，則引發錯誤

        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192  # 每次讀取的塊大小
        
        start_time = time.time()
        downloaded_size = 0

        with open(local_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.split('/')[-1]) as pbar:
                for data in response.iter_content(chunk_size=chunk_size):
                    downloaded_size += len(data)
                    f.write(data)
                    pbar.update(len(data))
                    
                    # 速度控制邏輯
                    elapsed_time = time.time() - start_time
                    current_speed = downloaded_size / elapsed_time if elapsed_time > 0 else 0
                    
                    if current_speed > speed_limit:
                        # 如果速度超過上限，就暫停一下
                        sleep_time = (downloaded_size / speed_limit) - elapsed_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)

    except Exception as e:
        print(f"下載失敗：{e}")
        return False
    return True

# 這是模型檔案的清單，你需要根據實際情況修改
# 通常你可以去 Hugging Face 模型頁面的 "Files and versions" 頁籤找到這些檔案
files_to_download = [
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "model-00001-of-0000x.safetensors", # 這需要根據模型實際分割的檔案數來修改
    "model.safetensors.index.json",
    "generation_config.json"
]

# 假設這些檔案都位於模型庫的 LFS 伺服器
# 你需要找到正確的檔案 URL
base_url = f"https://huggingface.co/{model_id}/resolve/main/"
for file_name in files_to_download:
    url = base_url + file_name
    local_path = os.path.join(local_dir, file_name)
    download_file_with_limit(url, local_path, download_speed_limit)

# 從本地路徑載入模型和分詞器
tokenizer = AutoTokenizer.from_pretrained(local_dir)
model = Llama4ForConditionalGeneration.from_pretrained(
    local_dir,
    tp_plan="auto",
    torch_dtype="auto",
)

# 執行推理
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])