"""
SQL Database Setup for Social Media and Mental Health Analysis
This script creates the database structure and imports processed data.
"""

import os
import sqlite3
import pandas as pd
import json
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
DB_PATH = os.path.join(DATA_DIR, 'social_media_mental_health.db')

def create_database():
    """
    Create SQLite database with tables for tweets, sentiment, and user metrics
    
    Returns:
        sqlite3.Connection: Database connection
    """
    print(f"Creating database at {DB_PATH}")
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tweets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tweets (
        tweet_id TEXT PRIMARY KEY,
        tweet_text TEXT,
        cleaned_text TEXT,
        created_at TEXT,
        collected_at TEXT,
        search_keyword TEXT,
        username TEXT,
        retweet_count INTEGER,
        favorite_count INTEGER
    )
    ''')
    
    # Create tweet_sentiment table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tweet_sentiment (
        tweet_id TEXT PRIMARY KEY,
        textblob_polarity REAL,
        textblob_subjectivity REAL,
        textblob_category TEXT,
        vader_compound REAL,
        vader_positive REAL,
        vader_negative REAL,
        vader_neutral REAL,
        vader_category TEXT,
        contains_mental_health_term INTEGER,
        mental_health_terms TEXT,
        FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id)
    )
    ''')
    
    # Create user_metrics table (aggregated, anonymized)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_metrics (
        username TEXT PRIMARY KEY,
        tweet_count INTEGER,
        avg_textblob_polarity REAL,
        avg_vader_compound REAL,
        mental_health_tweet_count INTEGER,
        avg_engagement REAL
    )
    ''')
    
    # Create mental_health_terms table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mental_health_terms (
        term_id INTEGER PRIMARY KEY AUTOINCREMENT,
        term_text TEXT UNIQUE,
        category TEXT,
        occurrence_count INTEGER DEFAULT 0
    )
    ''')
    
    # Commit changes and return connection
    conn.commit()
    return conn

def import_processed_data(conn, csv_file):
    """
    Import processed data from CSV file into database
    
    Args:
        conn (sqlite3.Connection): Database connection
        csv_file (str): Path to CSV file
        
    Returns:
        int: Number of records imported
    """
    print(f"Importing data from {csv_file}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Clean up data for import
        if 'id' in df.columns:
            df.rename(columns={'id': 'tweet_id'}, inplace=True)
            
        # Handle missing columns
        required_columns = ['tweet_id', 'text', 'cleaned_text', 'created_at', 'collected_at', 
                           'search_keyword', 'textblob_polarity', 'textblob_subjectivity', 
                           'textblob_category', 'vader_compound', 'vader_positive', 
                           'vader_negative', 'vader_neutral', 'vader_category', 
                           'mental_health_terms', 'contains_mental_health_term']
                           
        for col in required_columns:
            if col not in df.columns:
                if col == 'tweet_id':
                    df['tweet_id'] = df.index.astype(str)
                elif col in ['created_at', 'collected_at']:
                    df[col] = datetime.now().isoformat()
                elif col == 'search_keyword':
                    df[col] = 'unknown'
                elif col in ['textblob_polarity', 'textblob_subjectivity', 'vader_compound', 
                            'vader_positive', 'vader_negative', 'vader_neutral']:
                    df[col] = 0.0
                elif col in ['textblob_category', 'vader_category']:
                    df[col] = 'neutral'
                elif col == 'mental_health_terms':
                    df[col] = df[col].apply(lambda x: '[]' if pd.isna(x) else x)
                elif col == 'contains_mental_health_term':
                    df[col] = 0
                else:
                    df[col] = ''
        
        # Convert mental_health_terms from string to list if needed
        if 'mental_health_terms' in df.columns:
            df['mental_health_terms'] = df['mental_health_terms'].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else 
                         (json.dumps(eval(x)) if isinstance(x, str) and x.startswith('[') else json.dumps([]))
            )
        
        # Extract username from user column if available
        if 'user' in df.columns and 'username' not in df.columns:
            try:
                df['username'] = df['user'].apply(
                    lambda x: x.get('username', '') if isinstance(x, dict) else 
                             (json.loads(x).get('username', '') if isinstance(x, str) else '')
                )
            except:
                df['username'] = ''
        
        # Import tweets
        tweets_df = df[['tweet_id', 'text', 'cleaned_text', 'created_at', 
                       'collected_at', 'search_keyword', 'username']].copy()
                       
        # Add engagement metrics if available
        if 'retweet_count' in df.columns:
            tweets_df['retweet_count'] = df['retweet_count']
        else:
            tweets_df['retweet_count'] = 0
            
        if 'favorite_count' in df.columns:
            tweets_df['favorite_count'] = df['favorite_count']
        else:
            tweets_df['favorite_count'] = 0
            
        tweets_df.to_sql('tweets', conn, if_exists='append', index=False)
        
        # Import sentiment data
        sentiment_df = df[['tweet_id', 'textblob_polarity', 'textblob_subjectivity', 
                          'textblob_category', 'vader_compound', 'vader_positive', 
                          'vader_negative', 'vader_neutral', 'vader_category', 
                          'contains_mental_health_term', 'mental_health_terms']].copy()
        sentiment_df.to_sql('tweet_sentiment', conn, if_exists='append', index=False)
        
        # Update mental health terms table
        cursor = conn.cursor()
        
        # Extract all mental health terms
        all_terms = []
        for terms_json in df['mental_health_terms']:
            try:
                terms = json.loads(terms_json) if isinstance(terms_json, str) else terms_json
                if isinstance(terms, list):
                    all_terms.extend(terms)
            except:
                pass
                
        # Count term occurrences
        term_counts = {}
        for term in all_terms:
            term_counts[term] = term_counts.get(term, 0) + 1
            
        # Update mental_health_terms table
        for term, count in term_counts.items():
            cursor.execute('''
            INSERT INTO mental_health_terms (term_text, category, occurrence_count)
            VALUES (?, ?, ?)
            ON CONFLICT(term_text) DO UPDATE SET
            occurrence_count = occurrence_count + ?
            ''', (term, 'general', count, count))
            
        # Calculate and import user metrics
        user_metrics = df.groupby('username').agg({
            'tweet_id': 'count',
            'textblob_polarity': 'mean',
            'vader_compound': 'mean',
            'contains_mental_health_term': 'sum',
            'retweet_count': 'mean',
            'favorite_count': 'mean'
        }).reset_index()
        
        user_metrics.rename(columns={
            'tweet_id': 'tweet_count',
            'textblob_polarity': 'avg_textblob_polarity',
            'vader_compound': 'avg_vader_compound',
            'contains_mental_health_term': 'mental_health_tweet_count'
        }, inplace=True)
        
        # Calculate average engagement
        user_metrics['avg_engagement'] = (user_metrics['retweet_count'] + user_metrics['favorite_count']) / 2
        
        # Drop temporary columns
        user_metrics.drop(['retweet_count', 'favorite_count'], axis=1, inplace=True)
        
        # Import user metrics
        user_metrics.to_sql('user_metrics', conn, if_exists='append', index=False)
        
        # Commit changes
        conn.commit()
        
        return len(df)
        
    except Exception as e:
        print(f"Error importing data: {e}")
        return 0

def import_all_processed_data(conn, processed_dir=None):
    """
    Import all processed CSV files into database
    
    Args:
        conn (sqlite3.Connection): Database connection
        processed_dir (str): Directory containing processed CSV files
        
    Returns:
        int: Total number of records imported
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR
        
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Find all processed CSV files
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') and 'processed' in f]
    
    if not csv_files:
        print(f"No processed CSV files found in {processed_dir}")
        return 0
        
    total_imported = 0
    
    for file in csv_files:
        file_path = os.path.join(processed_dir, file)
        imported = import_processed_data(conn, file_path)
        total_imported += imported
        print(f"Imported {imported} records from {file}")
        
    return total_imported

def main():
    """Main function to set up database and import data"""
    print(f"Starting database setup at {datetime.now().isoformat()}")
    
    # Create database
    conn = create_database()
    
    # Import all processed data
    total_imported = import_all_processed_data(conn)
    
    print(f"Total records imported: {total_imported}")
    
    # Close connection
    conn.close()
    
    print(f"Database setup completed at {datetime.now().isoformat()}")
    print(f"Database path: {DB_PATH}")

if __name__ == "__main__":
    main()
