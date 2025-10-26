import pandas as pd
from transformers import pipeline
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings('ignore')


def analyze_sentiment(text_list):
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("Sentiment analysis pipeline loaded.")
    except Exception as e:
        print(f"Error loading sentiment pipeline: {e}")
        print("Make sure you have an internet connection and the transformers library installed.")
        print("You might need to install PyTorch or TensorFlow: pip install torch or pip install tensorflow")
        return None

    try:
        results = sentiment_pipeline(text_list)
        scores = []
        for result in results:
            label = result.get('label', 'UNKNOWN')
            score_value = result.get('score', 0.0)
            if label == 'POSITIVE':
                scores.append(1.0)
            elif label == 'NEGATIVE':
                scores.append(-1.0)
            else:
                 scores.append(0.0)
        return results, scores
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None, None


if __name__ == "__main__":
    sample_headlines = [
        "Oilseed prices surge due to unexpected export demand.",
        "Government announces reduction in import duties for palm oil, impacting local prices.",
        "Weather remains stable in key growing regions, harvest outlook normal.",
        "Global soybean futures drop sharply on supply glut fears.",
        "Farmers protest low MSP, market arrivals slow down."
    ]

    print("Sample Headlines:")
    for i, headline in enumerate(sample_headlines):
        print(f"{i+1}. {headline}")

    print("\nRunning sentiment analysis...")
    analysis_results, sentiment_scores = analyze_sentiment(sample_headlines)

    if analysis_results and sentiment_scores:
        print("\nSentiment Analysis Results:")
        df_results = pd.DataFrame({
            'Headline': sample_headlines,
            'Predicted Label': [res.get('label') for res in analysis_results],
            'Confidence': [res.get('score') for res in analysis_results],
            'Mapped Score': sentiment_scores
        })
        print(df_results)

        if sentiment_scores:
            average_score = sum(sentiment_scores) / len(sentiment_scores)
            print(f"\nAverage Sentiment Score for these headlines: {average_score:.4f}")
        else:
            print("\nCould not calculate average score.")
    else:
        print("\nSentiment analysis failed.")

