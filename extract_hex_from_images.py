import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import os

# Initialize model and processor
# checkpoint = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
# checkpoint = "HuggingFaceTB/SmolVLM2-250M-Video-Instruct"
checkpoint = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
print(f"Loading {checkpoint}...")

try:
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    print(f"Error loading model with AutoModelForImageTextToText: {e}")
    print("Trying with AutoModelForVision2Seq...")
    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, 
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"Error loading model with AutoModelForVision2Seq: {e}")
        exit(1)

# Directory containing images
img_dir = "./img/"
prompt_text = "Extract any hexadecimal numbers present in the image."

print("\n" + "="*50)
print(f"Running Hex Extraction for images in {img_dir}")
print("="*50 + "\n")

# Get list of image files
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
image_files.sort()

for image_file in image_files:
    image_path = os.path.join(img_dir, image_file)
    print(f"Processing: {image_file}")
    
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Could not load image {image_file}: {e}")
        continue

    messages = [
        {"role": "system", "content": [{"type":"text","text":"You are a helpful assistant that processes images and text. "}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    
    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        
        # Generate response
        outputs = model.generate(**inputs, max_new_tokens=100)
        
        # Decode
        response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"Result for {image_file}: {response.strip()}")
    except Exception as e:
        print(f"Error during generation for {image_file}: {e}")
        # Fallback
        try:
            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            response = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"Result for {image_file} (Fallback): {response.strip()}")
        except Exception as e2:
            print(f"Fallback failed for {image_file}: {e2}")

    print("-" * 30)
