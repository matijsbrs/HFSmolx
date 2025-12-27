import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize model and tokenizer
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
print(f"Loading {checkpoint}...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to("cuda" if torch.cuda.is_available() else "cpu")

# Define a list of prompts to showcase specific capabilities
prompts = [
    {
        "category": "1. Simple Knowledge & Explanation",
        "system": "You are a helpful assistant.",
        "user": "Explain what a black hole is to a 5-year-old."
    },
    {
        "category": "2. Creative Writing (Short Form)",
        "system": "You are a creative poet.",
        "user": "Write a haiku about a rainy day."
    },
    {
        "category": "3. Instruction Following (Lists)",
        "system": "You are a helpful assistant.",
        "user": "List 3 tips for staying focused while studying."
    },
    {
        "category": "4. Text Rewriting / Style Transfer",
        "system": "You are a professional editor.",
        "user": "Rewrite this sentence to be more polite and professional: 'I want the report now, it's late.'"
    },
    {
        "category": "5. Basic Coding",
        "system": "You are a Python coding assistant.",
        "user": "Write a simple Python function that returns the square of a number."
    }
]

print("\n" + "="*50)
print(f"Running Showcase for {checkpoint}")
print("="*50 + "\n")

for item in prompts:
    print(f"--- Category: {item['category']} ---")
    print(f"User: {item['user']}")
    
    messages = [
        {"role": "system", "content": item['system']},
        {"role": "user", "content": item['user']},
    ]
    
    # Manual tokenization to avoid version compatibility issues
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        temperature=0.6,  # Slightly creative but stable
        top_p=0.9,
        do_sample=True
    )
    
    # Decode and print only the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"SmolLM2: {response.strip()}")
    print("\n" + "-"*30 + "\n")
