import pandas as pd

# Define the data with diverse sentiments
data = {
    "text": [
        "I absolutely love this product! It's amazing.",
        "I hate this service. It's the worst I've ever experienced.",
        "The food was okay, not great but not terrible either.",
        "I'm extremely happy with my purchase! Highly recommend.",
        "I will never buy from this company again.",
        "This is the best book I've read this year!",
        "The movie was a total waste of time.",
        "I'm indifferent about this product. It’s just fine.",
        "The customer service was fantastic and very helpful.",
        "I had high expectations, but it didn't meet them.",
        "The delivery was late, but the product is good.",
        "I'm thrilled with my new phone! The features are incredible.",
        "It broke after one use. Very disappointed.",
        "I love the design, but the functionality is lacking.",
        "This app makes my life so much easier!",
        "Not what I expected. The quality is subpar.",
        "I think it’s just average, nothing special.",
        "I can't believe how great this was!",
        "Terrible experience. I want a refund!",
        "This is exactly what I was looking for, perfect!",
        "I would give it zero stars if I could."
    ],
    "label": [
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sentiment_data.csv', index=False)

print("sentiment_data.csv has been created successfully!")
