"""
Twitter Data Collector for Social Media and Mental Health Analysis
This script collects tweets related to mental health topics using the Twitter API.
"""

import sys
import os
import json
import time
import pandas as pd
from datetime import datetime

# Add path for data API access
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

# Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mental health related keywords
MENTAL_HEALTH_KEYWORDS = [
    "mental health", "anxiety", "depression", "stress", "therapy", 
    "self care", "mindfulness", "burnout", "mental wellbeing",
    "mental illness", "panic attack", "insomnia", "mental wellness"
]

# Initialize API client
client = ApiClient()

def search_tweets(query, count=20, tweet_type="Top", cursor=None):
    """
    Search for tweets using the Twitter API
    
    Args:
        query (str): Search query
        count (int): Number of tweets to return
        tweet_type (str): Type of tweets to return (Top, Latest, etc.)
        cursor (str): Pagination cursor
        
    Returns:
        dict: API response
    """
    try:
        params = {
            'query': query,
            'count': count,
            'type': tweet_type
        }
        
        if cursor:
            params['cursor'] = cursor
            
        response = client.call_api('Twitter/search_twitter', query=params)
        return response
    except Exception as e:
        print(f"Error searching tweets: {e}")
        return None

def collect_mental_health_tweets(keywords, tweets_per_keyword=50, output_file=None):
    """
    Collect tweets related to mental health keywords
    
    Args:
        keywords (list): List of keywords to search for
        tweets_per_keyword (int): Number of tweets to collect per keyword
        output_file (str): Path to output file
        
    Returns:
        list: Collected tweets
    """
    all_tweets = []
    
    for keyword in keywords:
        print(f"Collecting tweets for keyword: {keyword}")
        collected = 0
        cursor = None
        
        while collected < tweets_per_keyword:
            response = search_tweets(keyword, count=20, tweet_type="Latest", cursor=cursor)
            
            if not response or 'result' not in response:
                print(f"No results found for keyword: {keyword}")
                break
                
            # Extract tweets from response
            try:
                timeline = response['result']['timeline']
                instructions = timeline.get('instructions', [])
                
                for instruction in instructions:
                    if 'entries' in instruction:
                        for entry in instruction['entries']:
                            if 'content' in entry and 'entryType' in entry['content']:
                                # Process tweet content
                                tweet_data = extract_tweet_data(entry['content'])
                                if tweet_data:
                                    tweet_data['search_keyword'] = keyword
                                    all_tweets.append(tweet_data)
                                    collected += 1
                                    
                                    if collected >= tweets_per_keyword:
                                        break
            except Exception as e:
                print(f"Error processing tweets: {e}")
                
            # Check if we have a cursor for pagination
            if 'cursor' in response and 'bottom' in response['cursor']:
                cursor = response['cursor']['bottom']
            else:
                break
                
            # Respect rate limits
            time.sleep(1)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_tweets, f, indent=2)
            
    return all_tweets

def extract_tweet_data(content):
    """
    Extract relevant data from tweet content
    
    Args:
        content (dict): Tweet content from API response
        
    Returns:
        dict: Extracted tweet data
    """
    try:
        # This is a simplified extraction - would need to be adapted to actual API response structure
        if 'items' in content:
            for item in content['items']:
                if 'item' in item and 'itemContent' in item['item']:
                    item_content = item['item']['itemContent']
                    
                    # Extract user data if available
                    user_data = {}
                    if 'user_results' in item_content and 'result' in item_content['user_results']:
                        user = item_content['user_results']['result']
                        if 'legacy' in user:
                            user_legacy = user['legacy']
                            user_data = {
                                'username': user_legacy.get('screen_name', ''),
                                'name': user_legacy.get('name', ''),
                                'followers_count': user_legacy.get('followers_count', 0),
                                'friends_count': user_legacy.get('friends_count', 0)
                            }
                    
                    # Extract tweet content
                    tweet_data = {
                        'id': item.get('entryId', ''),
                        'text': item_content.get('text', ''),
                        'created_at': item_content.get('created_at', ''),
                        'user': user_data,
                        'retweet_count': item_content.get('retweet_count', 0),
                        'favorite_count': item_content.get('favorite_count', 0),
                        'collected_at': datetime.now().isoformat()
                    }
                    
                    return tweet_data
    except Exception as e:
        print(f"Error extracting tweet data: {e}")
        
    return None

def main():
    """Main function to collect tweets"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"mental_health_tweets_{timestamp}.json")
    
    print(f"Starting tweet collection at {datetime.now().isoformat()}")
    print(f"Collecting tweets for {len(MENTAL_HEALTH_KEYWORDS)} keywords")
    
    tweets = collect_mental_health_tweets(
        MENTAL_HEALTH_KEYWORDS, 
        tweets_per_keyword=30,
        output_file=output_file
    )
    
    print(f"Collected {len(tweets)} tweets")
    print(f"Data saved to {output_file}")
    
    # Convert to DataFrame for easier analysis
    if tweets:
        df = pd.DataFrame(tweets)
        csv_file = os.path.join(OUTPUT_DIR, f"mental_health_tweets_{timestamp}.csv")
        df.to_csv(csv_file, index=False)
        print(f"CSV data saved to {csv_file}")

if __name__ == "__main__":
    main()
