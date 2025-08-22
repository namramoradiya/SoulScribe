import pandas as pd
import json
import os


df = pd.read_csv("datasets/emotion.csv")

df.dropna(subset=["Situation", "empathetic_dialogues"], inplace=True)

# Create a new DataFrame for training
df_processed = pd.DataFrame()
df_processed["input_text"] = df["Situation"].astype(str) + " " + df["empathetic_dialogues"].astype(str)
df_processed["target_text"] = df["empathetic_dialogues"].apply(lambda x: "I'm here for you. " + str(x))

# Save to JSON Lines format
os.makedirs("processed", exist_ok=True)
with open("processed/processed_dataset.json", "w", encoding="utf-8") as f:
    for i in range(len(df_processed)):
        json.dump({
            "input": df_processed.iloc[i]["input_text"],
            "output": df_processed.iloc[i]["target_text"]
        }, f)
        f.write("\n")

print("✅ Preprocessing complete. File saved as: processed/processed_dataset.json")
