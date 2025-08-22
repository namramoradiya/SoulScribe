from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/gpt-neo-125M"  # or your fine-tuned model repo
save_path = r"E:/SoulScribe/models/soulscribe_gptneo"

print("Downloading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Saving to local folder...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Model saved in", save_path)
print("You can now use this model in your generate_response.py script.")