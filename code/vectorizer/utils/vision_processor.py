import os
import logging
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image

# Set PyTorch memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class VisionProcessor:
    """
    Handles multimodal document processing (text extraction from images).
    """
    def __init__(self, model_id='google/gemma-3-12b-it', device="cuda:0"):
        self.device = device
        
        logging.info(f"Loading multimodal model: {model_id}...")
        try:
            # Use the correct model class for multimodal tasks
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map='auto',
            ).eval()
            
            # Use the correct processor for handling both text and images
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            
            logging.info("Multimodal model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading multimodal model: {e}")
            self.model = None
            self.processor = None

    def extract_text_with_vision(self, file_path: str) -> str:
        """
        Extracts text content from a visual file (image) using the model.
        """
        if not self.model or not self.processor:
            logging.error("Model or processor not loaded, cannot perform inference.")
            return ""

        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""

        try:
            image = Image.open(file_path).convert("RGB")
            
            # 將圖片轉換為灰階
            gray_image = image.convert('L')
            
            # 進行二值化處理 (這裡使用固定的閾值 128)
            # 大於 128 的像素點設為 255 (白色), 小於或等於 128 的設為 0 (黑色)
            binarized_image = gray_image.point(lambda p: p > 128 and 255)
            
            # Define the prompt using the chat template format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": binarized_image},
                        {"type": "text", "text": "請詳細描述這個文件的所有文字內容，包括標題、段落、表格、圖表說明等。請按照原始格式盡可能完整地轉錄所有可見的文字。"}
                    ]
                }
            ]
            
            # Prepare inputs using the processor's chat template function
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=self.model.dtype)

            input_len = inputs["input_ids"].shape[-1]
            
            # Perform inference
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,  # Set to True for more creative answers
                )
                
            # Decode the generated tokens
            output_tokens = generation[0][input_len:]
            decoded_text = self.processor.decode(output_tokens, skip_special_tokens=True)
            
            if decoded_text:
                logging.info(f"Successfully extracted text, length: {len(decoded_text)} characters.")
            else:
                logging.warning(f"Failed to extract any text content from the file: {file_path}")
            
            return decoded_text
        
        except Exception as e:
            logging.error(f"An error occurred while processing the file {file_path}: {e}", exc_info=True)
            return ""

# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    processor = VisionProcessor()

    # Create a dummy image file for demonstration
    from PIL import Image, ImageDraw, ImageFont
    dummy_image_path = "dummy_document.png"
    img = Image.new('RGB', (800, 600), color = (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()
    
    d.text((50,50), "This is a document title.", fill=(0,0,0), font=font)
    d.text((50,150), "Paragraph 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus sit amet odio ut lorem.", fill=(0,0,0), font=font)
    img.save(dummy_image_path)
    
    # Process the dummy image
    extracted_text = processor.extract_text_with_vision(dummy_image_path)
    print("\n--- Extracted Text ---")
    print(extracted_text)
    
    # Clean up the dummy file
    os.remove(dummy_image_path)