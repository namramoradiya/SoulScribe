from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/gpt-neo-125M"
save_path = "E:/SoulScribe/models/soluscribe_gptneo"

print("⬇ Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
print("⬇ Downloading model (this includes pytorch_model.bin)...")
model = AutoModelForCausalLM.from_pretrained(model_name, force_download=True)

print("💾 Saving to:", save_path)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Model and tokenizer saved successfully!")
