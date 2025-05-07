"""
Sentiment Analysis for Social Media and Mental Health
This script processes tweets and performs sentiment analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Create data directory if it doesn't exist
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Download necessary NLTK resources
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("NLTK resources may not have downloaded properly. Continuing anyway.")

def clean_tweet(tweet):
    """
    Clean tweet text by removing links, special characters, etc.
    
    Args:
        tweet (str): Raw tweet text
        
    Returns:
        str: Cleaned tweet text
    """
    if not isinstance(tweet, str):
        return ""
        
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove hashtag symbol but keep the text
    tweet = re.sub(r'#', '', tweet)
    
    # Remove non-alphanumeric characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    
    # Remove extra whitespace
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sentiment scores
    """
    analysis = TextBlob(text)
    
    # TextBlob sentiment: polarity (-1 to 1) and subjectivity (0 to 1)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity,
        'sentiment_category': 'positive' if analysis.sentiment.polarity > 0 else 
                             ('negative' if analysis.sentiment.polarity < 0 else 'neutral')
    }

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Sentiment scores
    """
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    return {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'sentiment_category': 'positive' if scores['compound'] >= 0.05 else 
                             ('negative' if scores['compound'] <= -0.05 else 'neutral')
    }

def extract_mental_health_terms(text, mental_health_terms):
    """
    Extract mental health related terms from text
    
    Args:
        text (str): Text to analyze
        mental_health_terms (list): List of mental health terms to look for
        
    Returns:
        list: Found mental health terms
    """
    found_terms = []
    text_lower = text.lower()
    
    for term in mental_health_terms:
        if term.lower() in text_lower:
            found_terms.append(term)
            
    return found_terms

def process_tweet_file(file_path, output_dir=None):
    """
    Process a JSON file containing tweets
    
    Args:
        file_path (str): Path to JSON file
        output_dir (str): Directory to save processed data
        
    Returns:
        pd.DataFrame: Processed tweets
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR
        
    print(f"Processing file: {file_path}")
    
    try:
        # Load tweets
        with open(file_path, 'r') as f:
            tweets = json.load(f)
            
        if not tweets:
            print("No tweets found in file")
            return None
            
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(tweets)
        
        # Extract text from tweets if needed
        if 'text' not in df.columns and isinstance(df.iloc[0], dict):
            df['text'] = df.apply(lambda row: row.get('text', ''), axis=1)
            
        # Clean tweets
        df['cleaned_text'] = df['text'].apply(clean_tweet)
        
        # Analyze sentiment using TextBlob
        textblob_sentiments = df['cleaned_text'].apply(analyze_sentiment_textblob)
        df['textblob_polarity'] = textblob_sentiments.apply(lambda x: x['polarity'])
        df['textblob_subjectivity'] = textblob_sentiments.apply(lambda x: x['subjectivity'])
        df['textblob_category'] = textblob_sentiments.apply(lambda x: x['sentiment_category'])
        
        # Analyze sentiment using VADER
        vader_sentiments = df['cleaned_text'].apply(analyze_sentiment_vader)
        df['vader_compound'] = vader_sentiments.apply(lambda x: x['compound'])
        df['vader_positive'] = vader_sentiments.apply(lambda x: x['positive'])
        df['vader_negative'] = vader_sentiments.apply(lambda x: x['negative'])
        df['vader_neutral'] = vader_sentiments.apply(lambda x: x['neutral'])
        df['vader_category'] = vader_sentiments.apply(lambda x: x['sentiment_category'])
        
        # Extract mental health terms
        mental_health_terms = [
            "mental health", "anxiety", "depression", "stress", "therapy", 
            "self care", "mindfulness", "burnout", "mental wellbeing",
            "mental illness", "panic attack", "insomnia", "mental wellness"
        ]
        df['mental_health_terms'] = df['cleaned_text'].apply(
            lambda x: extract_mental_health_terms(x, mental_health_terms)
        )
        df['contains_mental_health_term'] = df['mental_health_terms'].apply(lambda x: len(x) > 0)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(file_path).split('.')[0]
        output_file = os.path.join(output_dir, f"{base_filename}_processed_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        
        print(f"Processed data saved to {output_file}")
        return df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def process_all_tweet_files(data_dir=None, output_dir=None):
    """
    Process all JSON files in a directory
    
    Args:
        data_dir (str): Directory containing JSON files
        output_dir (str): Directory to save processed data
        
    Returns:
        pd.DataFrame: Combined processed tweets
    """
    if data_dir is None:
        data_dir = DATA_DIR
        
    if output_dir is None:
        output_dir = PROCESSED_DIR
        
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.json') and 'mental_health_tweets' in f]
    
    if not all_files:
        print(f"No tweet files found in {data_dir}")
        return None
        
    all_dfs = []
    
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        df = process_tweet_file(file_path, output_dir)
        
        if df is not None:
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data processed")
        return None
        
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save combined data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_file = os.path.join(output_dir, f"all_tweets_processed_{timestamp}.csv")
    combined_df.to_csv(combined_file, index=False)
    
    print(f"Combined processed data saved to {combined_file}")
    return combined_df

def main():
    """Main function to process tweets"""
    print(f"Starting tweet processing at {datetime.now().isoformat()}")
    
    # Process all tweet files
    combined_df = process_all_tweet_files()
    
    if combined_df is not None:
        print(f"Processed {len(combined_df)} tweets")
        
        # Print sentiment distribution
        print("\nSentiment Distribution (TextBlob):")
        print(combined_df['textblob_category'].value_counts())
        
        print("\nSentiment Distribution (VADER):")
        print(combined_df['vader_category'].value_counts())
        
        # Print mental health term statistics
        print("\nTweets containing mental health terms:")
        print(combined_df['contains_mental_health_term'].value_counts())
        
        # Most common mental health terms
        all_terms = []
        for terms in combined_df['mental_health_terms']:
            all_terms.extend(terms)
            
        if all_terms:
            term_counts = pd.Series(all_terms).value_counts()
            print("\nMost common mental health terms:")
            print(term_counts.head(10))
    
if __name__ == "__main__":
    main()
