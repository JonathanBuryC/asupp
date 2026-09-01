from pathlib import Path
import sys
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# --- Setup imports ---
sys.path.append(str(Path().resolve().parent))

# =====================
# 1. Load datasets
# =====================
train_df = pd.read_excel("/opt/app-root/src/uc202-ipn-rex/notebooks/train_set.xlsx")
val_df = pd.read_excel("/opt/app-root/src/uc202-ipn-rex/notebooks/val_set.xlsx")
test_df = pd.read_excel("/opt/app-root/src/uc202-ipn-rex/notebooks/test_set.xlsx")

# Keep only positives (Annotation == 2)
train_pos = train_df[train_df["Annotation"] == 2]
val_pos = val_df[val_df["Annotation"] == 2]
test_pos = test_df[test_df["Annotation"] == 2]

# Build dicts: {Question: [list of positive answers]}
train_q2a = train_pos.groupby("Question")["Content"].apply(list).to_dict()
val_q2a = val_pos.groupby("Question")["Content"].apply(list).to_dict()


# =====================
# 2. Build InputExamples (old API style)
# =====================
def build_examples(question2answers, num_cycles=3):
    examples = []
    for _ in range(num_cycles):
        for q, answers in question2answers.items():
            for a in answers:
                if isinstance(q, str) and isinstance(a, str):
                    examples.append(InputExample(texts=[q.strip(), a.strip()]))
    return examples


train_examples = build_examples(train_q2a, num_cycles=3)
val_examples = build_examples(val_q2a, num_cycles=1)

print(f"Train examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")

# =====================
# 3. Load model
# =====================
model = SentenceTransformer(
    "/opt/app-root/src/uc202-ipn-rex/src/models/models/model_tigran",
    trust_remote_code=True,
)

# =====================
# 4. Define DataLoaders and Loss
# =====================
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=16)

train_loss = losses.MultipleNegativesRankingLoss(model)

# =====================
# 5. Training
# =====================
num_epochs = 5
batch_size = 32
warmup_ratio = 0.1  # 10% des steps pour le warmup
learning_rate = 3e-5
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=int(warmup_ratio * len(train_dataloader) * num_epochs),
    output_path="/opt/app-root/src/uc202-ipn-rex/src/models/models/model_tigran_finetuned",
    show_progress_bar=True,
    optimizer_params={"lr": learning_rate},
    weight_decay=0.01,
    scheduler="warmupcosine",
    checkpoint_path=None,  # ou un dossier si tu veux sauvegarder chaque epoch
    use_amp=True,  # active le float16 pour accélérer si GPU compatible
)
