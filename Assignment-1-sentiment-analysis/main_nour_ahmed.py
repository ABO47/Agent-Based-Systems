from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ─────────────────────────────────────────────
#  Section 1: Quick Start with Pipeline API
# ─────────────────────────────────────────────
print("=" * 60)
print("Method 1: Using Pipeline API")
print("=" * 60)

sa_pipe = pipeline("sentiment-analysis")

sample_a = "I absolutely love this product! It's amazing!"
output_a = sa_pipe(sample_a)
print(f"\nText: {sample_a}")
print(f"Result: {output_a[0]}")

sample_b = "This is the worst experience I've ever had."
output_b = sa_pipe(sample_b)
print(f"\nText: {sample_b}")
print(f"Result: {output_b[0]}")

# ─────────────────────────────────────────────
#  Section 2: Manual Inference with a Specific Model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Method 2: Using Specific Pre-trained Model")
print("=" * 60)

chosen_model = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(chosen_model)
clf = AutoModelForSequenceClassification.from_pretrained(chosen_model)

def get_sentiment(sentence):
    # Tokenize and prepare tensors
    encoded = tok(sentence, return_tensors="pt", truncation=True, padding=True)

    # Run inference without tracking gradients
    with torch.no_grad():
        raw_output = clf(**encoded)

    # Convert logits to probabilities
    scores = torch.nn.functional.softmax(raw_output.logits, dim=-1)

    # Pick the highest-scoring class
    top_class = torch.argmax(scores, dim=-1).item()
    top_score = scores[0][top_class].item()

    sentiment_label = "POSITIVE" if top_class == 1 else "NEGATIVE"

    return {
        "label": sentiment_label,
        "confidence": top_score,
        "positive_score": scores[0][1].item(),
        "negative_score": scores[0][0].item()
    }

sentences = [
    "This movie was fantastic! I loved every minute of it.",
    "The service was terrible and the food was cold.",
    "It was okay, nothing special but not bad either."
]

for sentence in sentences:
    analysis = get_sentiment(sentence)
    print(f"\nText: {sentence}")
    print(f"Sentiment: {analysis['label']}")
    print(f"Confidence: {analysis['confidence']:.4f}")
    print(f"Positive Score: {analysis['positive_score']:.4f}")
    print(f"Negative Score: {analysis['negative_score']:.4f}")

# ─────────────────────────────────────────────
#  Section 3: Processing a Batch of Reviews
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Method 3: Batch Processing Multiple Texts")
print("=" * 60)

bulk_pipe = pipeline("sentiment-analysis", model=chosen_model, device=-1)

customer_reviews = [
    "Excellent product, highly recommend!",
    "Waste of money, very disappointed.",
    "Good quality for the price.",
    "Not what I expected, returning it.",
    "Perfect! Exactly what I needed."
]

bulk_results = bulk_pipe(customer_reviews)

print("\nBatch Analysis Results:")
print("-" * 60)
for review, res in zip(customer_reviews, bulk_results):
    print(f"Review: {review}")
    print(f"  → {res['label']} (confidence: {res['score']:.4f})\n")

# ─────────────────────────────────────────────
#  Section 4: Side-by-Side Model Comparison
# ─────────────────────────────────────────────
print("=" * 60)
print("Method 4: Comparing Different Models")
print("=" * 60)

roberta_pipe = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

comparison_text = "I'm so happy with this purchase! Best decision ever! "

print(f"\nText: {comparison_text}\n")
print("DistilBERT result:", sa_pipe(comparison_text)[0])
print("RoBERTa result:", roberta_pipe(comparison_text)[0])

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)