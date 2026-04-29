from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

print("=" * 60)
print("Method 1: Using Pipeline API")
print("=" * 60)

sentiment_pipeline = pipeline("sentiment-analysis")

text1 = "I absolutely love this product! It's amazing!"
result1 = sentiment_pipeline(text1)
print(f"\nText: {text1}")
print(f"Result: {result1[0]}")

text2 = "This is the worst experience I've ever had."
result2 = sentiment_pipeline(text2)
print(f"\nText: {text2}")
print(f"Result: {result2[0]}")

print("\n" + "=" * 60)
print("Method 2: Using Specific Pre-trained Model")
print("=" * 60)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    predicted_class = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][predicted_class].item()
    
    label = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
    
    return {
        "label": label,
        "confidence": confidence,
        "positive_score": probs[0][1].item(),
        "negative_score": probs[0][0].item()
    }

test_texts = [
    "This movie was fantastic! I loved every minute of it.",
    "The service was terrible and the food was cold.",
    "It was okay, nothing special but not bad either."
]

for text in test_texts:
    result = analyze_sentiment(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Positive Score: {result['positive_score']:.4f}")
    print(f"Negative Score: {result['negative_score']:.4f}")

print("\n" + "=" * 60)
print("Method 3: Batch Processing Multiple Texts")
print("=" * 60)

batch_pipeline = pipeline("sentiment-analysis", model=model_name, device=-1)

reviews = [
    "Excellent product, highly recommend!",
    "Waste of money, very disappointed.",
    "Good quality for the price.",
    "Not what I expected, returning it.",
    "Perfect! Exactly what I needed."
]

batch_results = batch_pipeline(reviews)

print("\nBatch Analysis Results:")
print("-" * 60)
for review, result in zip(reviews, batch_results):
    print(f"Review: {review}")
    print(f"  → {result['label']} (confidence: {result['score']:.4f})\n")

print("=" * 60)
print("Method 4: Comparing Different Models")
print("=" * 60)

roberta_pipeline = pipeline("sentiment-analysis", 
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest")

test_text = "I'm so happy with this purchase! Best decision ever! 😊"

print(f"\nText: {test_text}\n")
print("DistilBERT result:", sentiment_pipeline(test_text)[0])
print("RoBERTa result:", roberta_pipeline(test_text)[0])

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
