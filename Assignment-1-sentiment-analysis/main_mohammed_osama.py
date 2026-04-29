from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

def separator(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

separator("Method 1: Quick Sentiment with Pipeline")

basic_analyzer = pipeline("sentiment-analysis")

samples = [
    "I absolutely love this product! It's amazing!",
    "This is the worst experience I've ever had."
]

for text in samples:
    result = basic_analyzer(text)[0]
    print(f"\nInput: {text}")
    print(f"Output: {result['label']} (score: {result['score']:.4f})")

separator("Method 2: Manual Model Inference")

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

def predict_sentiment(text):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        logits = model(**encoded).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    neg_score, pos_score = probabilities[0].tolist()
    label = "POSITIVE" if pos_score > neg_score else "NEGATIVE"

    return {
        "label": label,
        "confidence": max(pos_score, neg_score),
        "pos": pos_score,
        "neg": neg_score
    }

test_sentences = [
    "This movie was fantastic! I loved every minute of it.",
    "The service was terrible and the food was cold.",
    "It was okay, nothing special but not bad either."
]

for sentence in test_sentences:
    res = predict_sentiment(sentence)
    print(f"\nSentence: {sentence}")
    print(f"→ {res['label']} | Confidence: {res['confidence']:.4f}")
    print(f"   (+): {res['pos']:.4f} | (-): {res['neg']:.4f}")

separator("Method 3: Batch Processing")

batch_analyzer = pipeline("sentiment-analysis", model=model_id)

reviews = [
    "Excellent product, highly recommend!",
    "Waste of money, very disappointed.",
    "Good quality for the price.",
    "Not what I expected, returning it.",
    "Perfect! Exactly what I needed."
]

batch_outputs = batch_analyzer(reviews)

for text, out in zip(reviews, batch_outputs):
    print(f"{text}")
    print(f"→ {out['label']} ({out['score']:.4f})\n")

separator("Method 4: Compare Models")

alt_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

example_text = "I'm so happy with this purchase! Best decision ever! "

print(f"\nText: {example_text}\n")
print("DistilBERT:", basic_analyzer(example_text)[0])
print("RoBERTa  :", alt_analyzer(example_text)[0])

separator("All Done ")