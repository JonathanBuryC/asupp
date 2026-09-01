from src.models.model import Model
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from src.models.utils import load_and_prepare_data
from src.constants.model import S3_PATH_BASE, LOCAL_BASE, LOCAL_BASE_FINETUNED
from datasets import Dataset, Features, Sequence, Value
from typing import Optional, Dict, List
from src.processing.pde_ple import es, PDE



class ModelTrainer:
    def __init__(self, model: Model):
        self.model = model.model  # SentenceTransformer

    # ---------------------
    # FINETUNE
    # ---------------------
    def finetune(
        self,
        train_path,
        val_path,
        test_path,
        output_dir=LOCAL_BASE_FINETUNED,
        num_train_epochs=1,
        batch_size=16,
        upload_to_s3=False,
        finetuned_dir_s3=S3_PATH_BASE,
    ):
        if self.model is None:
            raise ValueError("Load or download a model before finetuning.")

        # Prepare data
        train_dataset, val_dataset, test_dataset = load_and_prepare_data(
            train_path, val_path, test_path
        )

        loss = MultipleNegativesRankingLoss(self.model)

        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_ratio=0.1,
            fp16=True,
            bf16=False,
            batch_sampler="no_duplicates",
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            run_name=f"{self._sanitize_model_name()}_finetune",
        )

        trainer = SentenceTransformerTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            loss=loss,
        )

        trainer.train()
        self.model.save(output_dir)
        print(f"Finetuned model saved locally to {output_dir}")


        if upload_to_s3:
            bucket_path = (
                f"{finetuned_dir_s3}/{self._sanitize_model_name()}_finetuned.pickle"
            )
            self.model.upload_to_s3(bucket_path=bucket_path)
            print(f"Finetuned model uploaded to {bucket_path}")

    # ---------------------
    # GRID SEARCH
    # ---------------------
    """     m = Model("intfloat/multilingual-e5-large")
    m.download()

    # Finetune once
    m.finetune("train.xlsx", "val.xlsx", "test.xlsx", upload_to_s3=True)

    # Or hyperparam search
    param_grid = {
        "num_train_epochs": [1, 2],
        "batch_size": [16, 32]
    }
    m.gridsearch("train.xlsx", "val.xlsx", "test.xlsx", param_grid) """

    #! check base output , probably s3 bcz a lot of candidates, models_candidates
    def gridsearch(
        self,
        train_path,
        val_path,
        test_path,
        param_grid: Dict[str, List],
        base_output="zz",
    ):
        from itertools import product

        # Expand param grid
        keys, values = zip(*param_grid.items())
        configs = [dict(zip(keys, v)) for v in product(*values)]

        best_score = -1
        best_config = None
        best_model_path = None

        for cfg in configs:
            print(f"Trying config: {cfg}")
            output_dir = f"{base_output}/{self._sanitize_model_name()}_{'_'.join([f'{k}{v}' for k, v in cfg.items()])}"

            self.finetune(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                output_dir=output_dir,
                num_train_epochs=cfg.get("num_train_epochs", 1),
                batch_size=cfg.get("batch_size", 16),
            )

            
            score = self._evaluate_on_val(output_dir, val_path)
            print(f"Score for {cfg}: {score}")

            if score > best_score:
                best_score = score
                best_config = cfg
                best_model_path = output_dir

        print(f"Best config: {best_config}, score={best_score}")
        print(f"Model saved at {best_model_path}")



def evaluate_model(model_path, index_name, test_path, k_values=[20, 50, 100]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)

    test_df = pd.read_excel(test_path)

    # Group by Question to get all candidate answers + labels
    q2answers = (
        test_df.groupby("Question")
        .apply(lambda x: list(zip(x["Content"], x["Annotation"])))
        .to_dict()
    )

    metrics = {k: 0 for k in k_values}
    total_qs = len(q2answers)

    for q, answers in q2answers.items():
        # encode query
        with torch.no_grad():
            query_embedding = model.encode(q, convert_to_numpy=True).tolist()

        # search top max_k
        max_k = max(k_values)
        query_body = {
            "size": max_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding},
                    },
                }
            },
        }

        response = es.search(index=index_name, body=query_body)
        retrieved_chunks = [
            hit["_source"]["chunk_content"] for hit in response["hits"]["hits"]
        ]

        # get all relevant answers for this question
        relevant_answers = set([a for a, label in answers if label == 1])

        for k in k_values:
            topk = set(retrieved_chunks[:k])
            if any(r in topk for r in relevant_answers):
                metrics[k] += 1

    # Normalize
    metrics = {k: v / total_qs for k, v in metrics.items()}
    return metrics
