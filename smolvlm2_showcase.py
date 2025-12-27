import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Initialize model and processor
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

# Load the local image
image_path = "demo2.jpg"
try:
    image = Image.open(image_path)
    print(f"Loaded image from {image_path}")
except Exception as e:
    print(f"Could not load image: {e}")
    exit(1)

# Define prompts
prompts = [
    # "Extract text from the image.",
    "Extract any hexadecimal numbers present in the image.",
    # "Describe the visual elements of the image."
]

print("\n" + "="*50)
print(f"Running Showcase for {checkpoint}")
print("="*50 + "\n")

for prompt_text in prompts:
    print(f"User: {prompt_text}")
    
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
    
    # Prepare inputs
    # Note: The API for apply_chat_template with images might vary. 
    # Some processors handle it directly, others need manual construction.
    # SmolVLM usually expects a specific format.
    
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
        print(f"SmolVLM2: {response.strip()}")
    except Exception as e:
        print(f"Error during generation: {e}")
        # Fallback for older processors that might not support apply_chat_template with images well
        try:
            print("Attempting fallback generation...")
            inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            response = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"SmolVLM2 (Fallback): {response.strip()}")
        except Exception as e2:
            print(f"Fallback failed: {e2}")

    print("\n" + "-"*30 + "\n")
