"""Optional offline Hugging Face judge for open-ended answers."""
from __future__ import annotations
import json

class LocalLLMJudge:
    def __init__(self, model_id: str):
        from transformers import pipeline
        self.model_id = model_id
        self.pipe = pipeline("text-generation", model=model_id, device_map="auto")
    def judge(self, question: str, reference: str, prediction: str) -> dict:
        prompt = ("You are an offline QA evaluator. Return JSON only with keys "
                  "score (0 or 1) and reason. Decide whether the prediction answers "
                  "the question consistently with the reference.\n"
                  f"Question: {question}\nReference: {reference}\nPrediction: {prediction}\n")
        raw = self.pipe(prompt, max_new_tokens=128, do_sample=False)[0]["generated_text"]
        tail = raw[len(prompt):].strip()
        try: result = json.loads(tail)
        except json.JSONDecodeError: result = {"score": 0, "reason": tail}
        result["model_id"] = self.model_id
        return result
