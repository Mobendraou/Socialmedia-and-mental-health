"""
Data Analysis for Social Media and Mental Health
This script performs analysis on the collected and processed data.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
DB_PATH = os.path.join(DATA_DIR, 'social_media_mental_health.db')
os.makedirs(RESULTS_DIR, exist_ok=True)

def connect_to_database():
    """
    Connect to the SQLite database
    
    Returns:
        sqlite3.Connection: Database connection
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    return sqlite3.connect(DB_PATH)

def analyze_sentiment_distribution(conn):
    """
    Analyze the distribution of sentiment in tweets
    
    Args:
        conn (sqlite3.Connection): Database connection
        
    Returns:
        pd.DataFrame: Sentiment distribution data
    """
    print("Analyzing sentiment distribution...")
    
    # Query sentiment distribution
    query = """
    SELECT 
        textblob_category, 
        vader_category,
        COUNT(*) as tweet_count
    FROM tweet_sentiment
    GROUP BY textblob_category, vader_category
    """
    
    sentiment_dist = pd.read_sql_query(query, conn)
    
    # Create separate distributions for TextBlob and VADER
    textblob_dist = sentiment_dist.groupby('textblob_category').agg({'tweet_count': 'sum'}).reset_index()
    vader_dist = sentiment_dist.groupby('vader_category').agg({'tweet_count': 'sum'}).reset_index()
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # TextBlob sentiment distribution
    sns.barplot(x='textblob_category', y='tweet_count', data=textblob_dist, ax=ax1)
    ax1.set_title('TextBlob Sentiment Distribution')
    ax1.set_xlabel('Sentiment Category')
    ax1.set_ylabel('Tweet Count')
    
    # VADER sentiment distribution
    sns.barplot(x='vader_category', y='tweet_count', data=vader_dist, ax=ax2)
    ax2.set_title('VADER Sentiment Distribution')
    ax2.set_xlabel('Sentiment Category')
    ax2.set_ylabel('Tweet Count')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'sentiment_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Sentiment distribution plot saved to {plot_path}")
    
    # Compare TextBlob and VADER agreement
    agreement_query = """
    SELECT 
        CASE WHEN textblob_category = vader_category THEN 'Agree' ELSE 'Disagree' END as agreement,
        COUNT(*) as count
    FROM tweet_sentiment
    GROUP BY agreement
    """
    
    agreement = pd.read_sql_query(agreement_query, conn)
    
    # Plot agreement
    plt.figure(figsize=(8, 6))
    sns.barplot(x='agreement', y='count', data=agreement)
    plt.title('Agreement Between TextBlob and VADER Sentiment Analysis')
    plt.xlabel('Agreement')
    plt.ylabel('Count')
    
    # Save plot
    agreement_path = os.path.join(RESULTS_DIR, 'sentiment_agreement.png')
    plt.savefig(agreement_path)
    plt.close()
    
    print(f"Sentiment agreement plot saved to {agreement_path}")
    
    return sentiment_dist

def analyze_mental_health_terms(conn):
    """
    Analyze the frequency of mental health terms in tweets
    
    Args:
        conn (sqlite3.Connection): Database connection
        
    Returns:
        pd.DataFrame: Mental health terms frequency data
    """
    print("Analyzing mental health terms frequency...")
    
    # Query mental health terms
    query = """
    SELECT term_text, occurrence_count
    FROM mental_health_terms
    ORDER BY occurrence_count DESC
    """
    
    terms_freq = pd.read_sql_query(query, conn)
    
    # Plot top 10 terms
    plt.figure(figsize=(10, 6))
    top_terms = terms_freq.head(10)
    sns.barplot(x='occurrence_count', y='term_text', data=top_terms)
    plt.title('Top 10 Mental Health Terms')
    plt.xlabel('Occurrence Count')
    plt.ylabel('Term')
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'mental_health_terms.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Mental health terms plot saved to {plot_path}")
    
    return terms_freq

def analyze_sentiment_by_term(conn):
    """
    Analyze sentiment distribution by mental health term
    
    Args:
        conn (sqlite3.Connection): Database connection
        
    Returns:
        pd.DataFrame: Sentiment by term data
    """
    print("Analyzing sentiment by mental health term...")
    
    # This query is complex because mental_health_terms is stored as JSON
    # We'll use Python to process this instead
    
    # Get tweets with sentiment and mental health terms
    query = """
    SELECT 
        t.tweet_id,
        t.cleaned_text,
        s.textblob_polarity,
        s.vader_compound,
        s.mental_health_terms
    FROM tweets t
    JOIN tweet_sentiment s ON t.tweet_id = s.tweet_id
    WHERE s.contains_mental_health_term = 1
    """
    
    tweets_df = pd.read_sql_query(query, conn)
    
    # Process mental health terms
    term_sentiment = []
    
    for _, row in tweets_df.iterrows():
        try:
            terms = json.loads(row['mental_health_terms'])
            for term in terms:
                term_sentiment.append({
                    'term': term,
                    'textblob_polarity': row['textblob_polarity'],
                    'vader_compound': row['vader_compound']
                })
        except:
            continue
    
    if not term_sentiment:
        print("No mental health terms found in tweets")
        return None
        
    term_sentiment_df = pd.DataFrame(term_sentiment)
    
    # Calculate average sentiment by term
    term_avg_sentiment = term_sentiment_df.groupby('term').agg({
        'textblob_polarity': 'mean',
        'vader_compound': 'mean',
        'term': 'count'
    }).rename(columns={'term': 'count'}).reset_index()
    
    # Filter to terms with at least 5 occurrences
    term_avg_sentiment = term_avg_sentiment[term_avg_sentiment['count'] >= 5]
    
    if term_avg_sentiment.empty:
        print("Not enough data to analyze sentiment by term")
        return term_sentiment_df
    
    # Sort by TextBlob polarity
    term_avg_sentiment = term_avg_sentiment.sort_values('textblob_polarity')
    
    # Plot sentiment by term
    plt.figure(figsize=(12, 8))
    
    # Create a horizontal bar chart
    sns.barplot(x='textblob_polarity', y='term', data=term_avg_sentiment)
    plt.title('Average Sentiment by Mental Health Term (TextBlob)')
    plt.xlabel('Average Sentiment Polarity')
    plt.ylabel('Mental Health Term')
    plt.axvline(x=0, color='gray', linestyle='--')
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'sentiment_by_term_textblob.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Sentiment by term plot (TextBlob) saved to {plot_path}")
    
    # Plot for VADER sentiment
    plt.figure(figsize=(12, 8))
    
    # Sort by VADER compound
    term_avg_sentiment = term_avg_sentiment.sort_values('vader_compound')
    
    # Create a horizontal bar chart
    sns.barplot(x='vader_compound', y='term', data=term_avg_sentiment)
    plt.title('Average Sentiment by Mental Health Term (VADER)')
    plt.xlabel('Average Sentiment Compound Score')
    plt.ylabel('Mental Health Term')
    plt.axvline(x=0, color='gray', linestyle='--')
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'sentiment_by_term_vader.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Sentiment by term plot (VADER) saved to {plot_path}")
    
    return term_avg_sentiment

def analyze_user_metrics(conn):
    """
    Analyze user metrics related to mental health tweets
    
    Args:
        conn (sqlite3.Connection): Database connection
        
    Returns:
        pd.DataFrame: User metrics data
    """
    print("Analyzing user metrics...")
    
    # Query user metrics
    query = """
    SELECT *
    FROM user_metrics
    WHERE tweet_count > 1
    """
    
    user_metrics = pd.read_sql_query(query, conn)
    
    if user_metrics.empty:
        print("No user metrics data available")
        return None
    
    # Calculate percentage of mental health tweets
    user_metrics['mental_health_percentage'] = (user_metrics['mental_health_tweet_count'] / 
                                              user_metrics['tweet_count']) * 100
    
    # Plot distribution of mental health tweet percentage
    plt.figure(figsize=(10, 6))
    sns.histplot(user_metrics['mental_health_percentage'], bins=20)
    plt.title('Distribution of Mental Health Tweet Percentage by User')
    plt.xlabel('Percentage of Tweets Related to Mental Health')
    plt.ylabel('Number of Users')
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'mental_health_tweet_percentage.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Mental health tweet percentage plot saved to {plot_path}")
    
    # Plot correlation between mental health tweet percentage and sentiment
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mental_health_percentage', y='avg_textblob_polarity', data=user_metrics)
    plt.title('Mental Health Tweet Percentage vs. Average Sentiment')
    plt.xlabel('Percentage of Tweets Related to Mental Health')
    plt.ylabel('Average Sentiment Polarity (TextBlob)')
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'mental_health_vs_sentiment.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Mental health vs sentiment plot saved to {plot_path}")
    
    # Plot correlation between engagement and sentiment
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='avg_engagement', y='avg_textblob_polarity', data=user_metrics)
    plt.title('Average Engagement vs. Average Sentiment')
    plt.xlabel('Average Engagement')
    plt.ylabel('Average Sentiment Polarity (TextBlob)')
    
    # Save plot
    plot_path = os.path.join(RESULTS_DIR, 'engagement_vs_sentiment.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Engagement vs sentiment plot saved to {plot_path}")
    
    return user_metrics

def generate_summary_report(sentiment_dist, terms_freq, term_sentiment, user_metrics):
    """
    Generate a summary report of the analysis
    
    Args:
        sentiment_dist (pd.DataFrame): Sentiment distribution data
        terms_freq (pd.DataFrame): Mental health terms frequency data
        term_sentiment (pd.DataFrame): Sentiment by term data
        user_metrics (pd.DataFrame): User metrics data
        
    Returns:
        str: Summary report
    """
    print("Generating summary report...")
    
    report = "# Social Media and Mental Health Analysis Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Sentiment distribution summary
    report += "## Sentiment Distribution\n\n"
    
    if sentiment_dist is not None:
        textblob_dist = sentiment_dist.groupby('textblob_category').agg({'tweet_count': 'sum'}).reset_index()
        vader_dist = sentiment_dist.groupby('vader_category').agg({'tweet_count': 'sum'}).reset_index()
        
        report += "### TextBlob Sentiment\n\n"
        report += textblob_dist.to_markdown() + "\n\n"
        
        report += "### VADER Sentiment\n\n"
        report += vader_dist.to_markdown() + "\n\n"
    else:
        report += "No sentiment distribution data available.\n\n"
    
    # Mental health terms summary
    report += "## Mental Health Terms\n\n"
    
    if terms_freq is not None and not terms_freq.empty:
        report += "### Top 10 Mental Health Terms\n\n"
        report += terms_freq.head(10).to_markdown() + "\n\n"
    else:
        report += "No mental health terms data available.\n\n"
    
    # Sentiment by term summary
    report += "## Sentiment by Mental Health Term\n\n"
    
    if term_sentiment is not None and not term_sentiment.empty:
        report += "### Average Sentiment by Term\n\n"
        report += term_sentiment.sort_values('textblob_polarity', ascending=False).head(10).to_markdown() + "\n\n"
    else:
        report += "No sentiment by term data available.\n\n"
    
    # User metrics summary
    report += "## User Metrics\n\n"
    
    if user_metrics is not None and not user_metrics.empty:
        # Calculate summary statistics
        avg_mental_health_pct = user_metrics['mental_health_percentage'].mean()
        avg_sentiment = user_metrics['avg_textblob_polarity'].mean()
        avg_engagement = user_metrics['avg_engagement'].mean()
        
        report += f"Average percentage of mental health tweets per user: {avg_mental_health_pct:.2f}%\n\n"
        report += f"Average sentiment polarity: {avg_sentiment:.4f}\n\n"
        report += f"Average engagement: {avg_engagement:.2f}\n\n"
    else:
        report += "No user metrics data available.\n\n"
    
    # Key findings
    report += "## Key Findings\n\n"
    
    # Add key findings based on available data
    findings = []
    
    if sentiment_dist is not None:
        textblob_dist = sentiment_dist.groupby('textblob_category').agg({'tweet_count': 'sum'}).reset_index()
        most_common = textblob_dist.loc[textblob_dist['tweet_count'].idxmax()]
        findings.append(f"The most common sentiment in mental health tweets is {most_common['textblob_category']}.")
    
    if terms_freq is not None and not terms_freq.empty:
        top_term = terms_freq.iloc[0]
        findings.append(f"The most frequently mentioned mental health term is '{top_term['term_text']}' with {top_term['occurrence_count']} occurrences.")
    
    if term_sentiment is not None and not term_sentiment.empty:
        most_positive = term_sentiment.loc[term_sentiment['textblob_polarity'].idxmax()]
        most_negative = term_sentiment.loc[term_sentiment['textblob_polarity'].idxmin()]
        findings.append(f"The mental health term with the most positive sentiment is '{most_positive['term']}' (polarity: {most_positive['textblob_polarity']:.4f}).")
        findings.append(f"The mental health term with the most negative sentiment is '{most_negative['term']}' (polarity: {most_negative['textblob_polarity']:.4f}).")
    
    if user_metrics is not None and not user_metrics.empty:
        # Check correlation between mental health tweet percentage and sentiment
        corr = user_metrics['mental_health_percentage'].corr(user_metrics['avg_textblob_polarity'])
        if abs(corr) > 0.3:
            direction = "positive" if corr > 0 else "negative"
            findings.append(f"There is a {direction} correlation ({corr:.4f}) between the percentage of mental health tweets and sentiment polarity.")
        
        # Check correlation between engagement and sentiment
        corr = user_metrics['avg_engagement'].corr(user_metrics['avg_textblob_polarity'])
        if abs(corr) > 0.3:
            direction = "positive" if corr > 0 else "negative"
            findings.append(f"There is a {direction} correlation ({corr:.4f}) between engagement and sentiment polarity.")
    
    if findings:
        for i, finding in enumerate(findings, 1):
            report += f"{i}. {finding}\n"
    else:
        report += "Insufficient data to generate key findings.\n"
    
    # Save report
    report_path = os.path.join(RESULTS_DIR, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report saved to {report_path}")
    
    return report

def main():
    """Main function to analyze data"""
    print(f"Starting data analysis at {datetime.now().isoformat()}")
    
    try:
        # Connect to database
        conn = connect_to_database()
        
        # Perform analyses
        sentiment_dist = analyze_sentiment_distribution(conn)
        terms_freq = analyze_mental_health_terms(conn)
        term_sentiment = analyze_sentiment_by_term(conn)
        user_metrics = analyze_user_metrics(conn)
        
        # Generate summary report
        generate_summary_report(sentiment_dist, terms_freq, term_sentiment, user_metrics)
        
        # Close connection
        conn.close()
        
        print(f"Data analysis completed at {datetime.now().isoformat()}")
        print(f"Results saved to {RESULTS_DIR}")
        
    except Exception as e:
        print(f"Error during data analysis: {e}")

if __name__ == "__main__":
    main()
