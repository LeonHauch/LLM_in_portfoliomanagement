# test_sentiment.py - Run this once to create test data
import pandas as pd

# Create dummy sentiment for your 2 assets
dates = pd.date_range('2024-01-01', periods=120, freq='D')
assets = ['ACA.PA', 'AI.PA']

sentiment_data = []
for date in dates:
    for asset in assets:
        sentiment_data.append({
            'date': date,
            'asset': asset,
            'sentiment': 0.5  # Dummy positive sentiment
        })

sentiment_df = pd.DataFrame(sentiment_data)
sentiment_df.to_csv('data/sentiment/test_sentiment.csv', index=False)

print(f"✅ Created test sentiment file with {len(sentiment_df)} rows")
print(sentiment_df.head())