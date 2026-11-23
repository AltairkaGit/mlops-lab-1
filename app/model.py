import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cointegrated/rubert-tiny-toxicity"
MODEL_REVISION = "5d37eff844868e243467e4c38898bad46c271af2"


class ToxicityClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
        ).eval()
        self.non_toxic_index = self.model.config.label2id["non-toxic"]
        self.dangerous_index = self.model.config.label2id["dangerous"]

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> list[float]:
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            return_tensors="pt",
        )
        scores = torch.sigmoid(self.model(**inputs).logits)
        non_toxic = scores[:, self.non_toxic_index]
        dangerous = scores[:, self.dangerous_index]
        return (1 - non_toxic * (1 - dangerous)).tolist()


def load_classifier() -> ToxicityClassifier:
    return ToxicityClassifier()
