import pandas as pd
from torch.utils.data import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from datasets import Dataset as HFDataset
from sentence_transformers.training_args import BatchSamplers

# ---- Custom Dataset that cycles through positives per question ----
class CyclicPositiveDataset(Dataset):
    def __init__(self, question2answers):
        self.questions = list(question2answers.keys())
        self.question2answers = question2answers
        # Track positions per question for cycling
        self.indices = {q: 0 for q in self.questions}
        self.column_names = "texts"

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q = self.questions[idx]
        answers = self.question2answers[q]

        # Get the current positive answer in the cycle
        pos_idx = self.indices[q]
        a = answers[pos_idx]

        # Move pointer to next positive (cyclically)
        self.indices[q] = (pos_idx + 1) % len(answers)

        # Return in the format expected by trainer
        return {"anchor": q, "positive": a}

    def print_epoch_pairs(self):
        """Print all current pairs (without advancing the cycle)"""
        for q in self.questions:
            idx = self.indices[q]
            a = self.question2answers[q][idx]
            print(f"Q: {q} --> A: {a}")

    def save_to_csv(self, file_path):
        """Save all question-answer pairs in a CSV suitable for trainer or inspection"""
        data = []
        for q, answers in self.question2answers.items():
            for a in answers:
                data.append({"anchor": q, "positive": a})
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False, columns=["anchor", "positive"])
        print(f"CSV saved to {file_path}")


# ---- Load your Excel datasets ----
train_df = pd.read_excel("train_set.xlsx")
val_df = pd.read_excel("val_set.xlsx")
test_df = pd.read_excel("test_set.xlsx")

# Keep only positive examples
train_pos = train_df[train_df["Annotation"] == 2]
val_pos = val_df[val_df["Annotation"] == 2]
test_pos = test_df[test_df["Annotation"] == 2]

# Build dictionaries: Question -> list of positive answers
train_q2a = train_pos.groupby("Question")["Content"].apply(list).to_dict()
val_q2a = val_pos.groupby("Question")["Content"].apply(list).to_dict()
test_q2a = test_pos.groupby("Question")["Content"].apply(list).to_dict()

# ---- Build cyclic datasets ----
train_dataset = CyclicPositiveDataset(train_q2a)
val_dataset = CyclicPositiveDataset(val_q2a)
test_dataset = CyclicPositiveDataset(test_q2a)
train_dataset.save_to_csv("epoch_1.xlsx")

# ---- Convert to HuggingFace Dataset format for the trainer ----
# Trainer expects a Dataset with `anchor` and `positive` columns
train_hf = HFDataset.from_dict(
    {
        "anchor": [train_dataset[i]["anchor"] for i in range(len(train_dataset))],
        "positive": [train_dataset[i]["positive"] for i in range(len(train_dataset))],
    }
)

val_hf = HFDataset.from_dict(
    {
        "anchor": [val_dataset[i]["anchor"] for i in range(len(val_dataset))],
        "positive": [val_dataset[i]["positive"] for i in range(len(val_dataset))],
    }
)

model = SentenceTransformer("intfloat/multilingual-e5-large", trust_remote_code=True)

# Define the loss
loss = MultipleNegativesRankingLoss(model)

#  Training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="./intfloat_multilingual_e5_large_instruct",
    num_train_epochs=20,
    per_device_train_batch_size=200,
    per_device_eval_batch_size=16,
    # warmup_ratio=0.1,
    fp16=True, 
    bf16=False,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    warmup_steps=50,
    eval_strategy="epoch",  
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    run_name="intfloat_multilingual_e5_large_instruct_test",
)

model.gradient_checkpointing_enable()

#  Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    loss=loss,
)

trainer.train()

# Save final model
model.save("./intfloat_multilingual_e5_large_instruct/test")
