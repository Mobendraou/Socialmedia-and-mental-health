"""
Dashboard Generator for Social Media and Mental Health Analysis
This script creates an interactive dashboard to visualize the analysis results.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
DASHBOARD_DIR = os.path.join(RESULTS_DIR, 'dashboard')
DB_PATH = os.path.join(DATA_DIR, 'social_media_mental_health.db')

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DASHBOARD_DIR, exist_ok=True)

def connect_to_database():
    """
    Connect to the SQLite database
    
    Returns:
        sqlite3.Connection: Database connection
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
        
    return sqlite3.connect(DB_PATH)

def load_data():
    """
    Load data from the database for dashboard visualization
    
    Returns:
        tuple: DataFrames containing tweets, sentiment, terms, and user metrics
    """
    conn = connect_to_database()
    
    # Load tweets with sentiment
    tweets_query = """
    SELECT 
        t.tweet_id,
        t.tweet_text,
        t.cleaned_text,
        t.created_at,
        t.search_keyword,
        t.username,
        t.retweet_count,
        t.favorite_count,
        s.textblob_polarity,
        s.textblob_subjectivity,
        s.textblob_category,
        s.vader_compound,
        s.vader_positive,
        s.vader_negative,
        s.vader_neutral,
        s.vader_category,
        s.contains_mental_health_term,
        s.mental_health_terms
    FROM tweets t
    JOIN tweet_sentiment s ON t.tweet_id = s.tweet_id
    """
    
    tweets_df = pd.read_sql_query(tweets_query, conn)
    
    # Load mental health terms
    terms_query = """
    SELECT *
    FROM mental_health_terms
    ORDER BY occurrence_count DESC
    """
    
    terms_df = pd.read_sql_query(terms_query, conn)
    
    # Load user metrics
    users_query = """
    SELECT *
    FROM user_metrics
    """
    
    users_df = pd.read_sql_query(users_query, conn)
    
    # Process mental health terms from JSON
    def extract_terms(terms_json):
        try:
            return json.loads(terms_json)
        except:
            return []
            
    tweets_df['extracted_terms'] = tweets_df['mental_health_terms'].apply(extract_terms)
    
    # Convert created_at to datetime
    try:
        tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'])
    except:
        pass
    
    conn.close()
    
    return tweets_df, terms_df, users_df

def create_dashboard():
    """
    Create an interactive Dash dashboard
    """
    # Load data
    try:
        tweets_df, terms_df, users_df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Define layout
    app.layout = html.Div([
        html.H1("Social Media and Mental Health Analysis Dashboard"),
        
        html.Div([
            html.H2("Overview"),
            html.P(f"Total tweets analyzed: {len(tweets_df)}"),
            html.P(f"Tweets containing mental health terms: {tweets_df['contains_mental_health_term'].sum()}"),
            html.P(f"Unique mental health terms identified: {len(terms_df)}"),
            html.P(f"Users analyzed: {len(users_df)}")
        ]),
        
        html.Div([
            html.H2("Sentiment Analysis"),
            
            # Sentiment distribution
            html.H3("Sentiment Distribution"),
            dcc.Graph(
                id='sentiment-distribution',
                figure=create_sentiment_distribution_figure(tweets_df)
            ),
            
            # Sentiment over time
            html.H3("Sentiment Over Time"),
            dcc.Graph(
                id='sentiment-time',
                figure=create_sentiment_time_figure(tweets_df)
            )
        ]),
        
        html.Div([
            html.H2("Mental Health Terms Analysis"),
            
            # Top mental health terms
            html.H3("Top Mental Health Terms"),
            dcc.Graph(
                id='top-terms',
                figure=create_top_terms_figure(terms_df)
            ),
            
            # Sentiment by term
            html.H3("Sentiment by Mental Health Term"),
            dcc.Graph(
                id='sentiment-by-term',
                figure=create_sentiment_by_term_figure(tweets_df)
            )
        ]),
        
        html.Div([
            html.H2("User Analysis"),
            
            # Mental health tweet percentage distribution
            html.H3("Mental Health Tweet Percentage Distribution"),
            dcc.Graph(
                id='mental-health-percentage',
                figure=create_mental_health_percentage_figure(users_df)
            ),
            
            # Engagement vs sentiment
            html.H3("Engagement vs Sentiment"),
            dcc.Graph(
                id='engagement-sentiment',
                figure=create_engagement_sentiment_figure(users_df)
            )
        ]),
        
        html.Div([
            html.H2("Tweet Explorer"),
            
            # Filters
            html.Div([
                html.Label("Filter by Sentiment:"),
                dcc.Dropdown(
                    id='sentiment-filter',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Positive', 'value': 'positive'},
                        {'label': 'Neutral', 'value': 'neutral'},
                        {'label': 'Negative', 'value': 'negative'}
                    ],
                    value='all'
                ),
                
                html.Label("Filter by Mental Health Term:"),
                dcc.Dropdown(
                    id='term-filter',
                    options=[{'label': 'All', 'value': 'all'}] + [
                        {'label': term, 'value': term} 
                        for term in terms_df['term_text'].head(20)
                    ],
                    value='all'
                )
            ]),
            
            # Tweet table
            html.Div(id='tweet-table')
        ]),
        
        html.Footer([
            html.P(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            html.P("Social Media and Mental Health Analysis Project")
        ])
    ])
    
    # Define callbacks
    @app.callback(
        Output('tweet-table', 'children'),
        [Input('sentiment-filter', 'value'),
         Input('term-filter', 'value')]
    )
    def update_tweet_table(sentiment, term):
        filtered_df = tweets_df.copy()
        
        # Filter by sentiment
        if sentiment != 'all':
            filtered_df = filtered_df[filtered_df['textblob_category'] == sentiment]
        
        # Filter by term
        if term != 'all':
            filtered_df = filtered_df[filtered_df['extracted_terms'].apply(lambda x: term in x if isinstance(x, list) else False)]
        
        # Limit to 20 tweets
        filtered_df = filtered_df.head(20)
        
        # Create table
        table_rows = []
        
        for _, row in filtered_df.iterrows():
            table_rows.append(html.Tr([
                html.Td(row['cleaned_text']),
                html.Td(row['textblob_category']),
                html.Td(f"{row['textblob_polarity']:.2f}"),
                html.Td(', '.join(row['extracted_terms']) if isinstance(row['extracted_terms'], list) else '')
            ]))
        
        return html.Table([
            html.Thead(
                html.Tr([
                    html.Th("Tweet Text"),
                    html.Th("Sentiment"),
                    html.Th("Polarity"),
                    html.Th("Mental Health Terms")
                ])
            ),
            html.Tbody(table_rows)
        ])
    
    # Save dashboard to HTML file
    app_html = app.index_string
    dashboard_path = os.path.join(DASHBOARD_DIR, 'dashboard.html')
    
    with open(dashboard_path, 'w') as f:
        f.write(app_html)
    
    print(f"Dashboard HTML saved to {dashboard_path}")
    
    # Return app for running
    return app

def create_sentiment_distribution_figure(tweets_df):
    """
    Create a figure showing sentiment distribution
    
    Args:
        tweets_df (pd.DataFrame): DataFrame containing tweets with sentiment
        
    Returns:
        plotly.graph_objects.Figure: Sentiment distribution figure
    """
    # Count sentiment categories
    textblob_counts = tweets_df['textblob_category'].value_counts().reset_index()
    textblob_counts.columns = ['category', 'count']
    
    vader_counts = tweets_df['vader_category'].value_counts().reset_index()
    vader_counts.columns = ['category', 'count']
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("TextBlob Sentiment", "VADER Sentiment"))
    
    # Add TextBlob sentiment bar chart
    fig.add_trace(
        go.Bar(
            x=textblob_counts['category'],
            y=textblob_counts['count'],
            name="TextBlob"
        ),
        row=1, col=1
    )
    
    # Add VADER sentiment bar chart
    fig.add_trace(
        go.Bar(
            x=vader_counts['category'],
            y=vader_counts['count'],
            name="VADER"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Sentiment Distribution",
        height=500
    )
    
    return fig

def create_sentiment_time_figure(tweets_df):
    """
    Create a figure showing sentiment over time
    
    Args:
        tweets_df (pd.DataFrame): DataFrame containing tweets with sentiment
        
    Returns:
        plotly.graph_objects.Figure: Sentiment over time figure
    """
    # Check if created_at is datetime
    if 'created_at' not in tweets_df.columns or not pd.api.types.is_datetime64_any_dtype(tweets_df['created_at']):
        # Create a dummy figure if time data is not available
        fig = go.Figure()
        fig.add_annotation(
            text="Time data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Sentiment Over Time (Data Not Available)",
            height=500
        )
        return fig
    
    # Group by date and calculate average sentiment
    tweets_df['date'] = tweets_df['created_at'].dt.date
    daily_sentiment = tweets_df.groupby('date').agg({
        'textblob_polarity': 'mean',
        'vader_compound': 'mean'
    }).reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add TextBlob sentiment line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['textblob_polarity'],
            mode='lines+markers',
            name="TextBlob Polarity"
        )
    )
    
    # Add VADER sentiment line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['vader_compound'],
            mode='lines+markers',
            name="VADER Compound"
        )
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=daily_sentiment['date'].min(),
        y0=0,
        x1=daily_sentiment['date'].max(),
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Average Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        height=500
    )
    
    return fig

def create_top_terms_figure(terms_df):
    """
    Create a figure showing top mental health terms
    
    Args:
        terms_df (pd.DataFrame): DataFrame containing mental health terms
        
    Returns:
        plotly.graph_objects.Figure: Top terms figure
    """
    # Get top 15 terms
    top_terms = terms_df.head(15)
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            y=top_terms['term_text'],
            x=top_terms['occurrence_count'],
            orientation='h'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Top Mental Health Terms",
        xaxis_title="Occurrence Count",
        yaxis_title="Term",
        height=600
    )
    
    return fig

def create_sentiment_by_term_figure(tweets_df):
    """
    Create a figure showing sentiment by mental health term
    
    Args:
        tweets_df (pd.DataFrame): DataFrame containing tweets with sentiment
        
    Returns:
        plotly.graph_objects.Figure: Sentiment by term figure
    """
    # Process mental health terms and sentiment
    term_sentiment = []
    
    for _, row in tweets_df.iterrows():
        terms = row['extracted_terms']
        if isinstance(terms, list) and terms:
            for term in terms:
                term_sentiment.append({
                    'term': term,
                    'textblob_polarity': row['textblob_polarity'],
                    'vader_compound': row['vader_compound']
                })
    
    if not term_sentiment:
        # Create a dummy figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No mental health terms data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Sentiment by Mental Health Term (Data Not Available)",
            height=500
        )
        return fig
    
    # Convert to DataFrame
    term_sentiment_df = pd.DataFrame(term_sentiment)
    
    # Calculate average sentiment by term
    term_avg = term_sentiment_df.groupby('term').agg({
        'textblob_polarity': 'mean',
        'vader_compound': 'mean',
        'term': 'count'
    }).rename(columns={'term': 'count'}).reset_index()
    
    # Filter to terms with at least 3 occurrences
    term_avg = term_avg[term_avg['count'] >= 3]
    
    if term_avg.empty:
        # Create a dummy figure if no data after filtering
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient mental health terms data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Sentiment by Mental Health Term (Insufficient Data)",
            height=500
        )
        return fig
    
    # Sort by TextBlob polarity
    term_avg = term_avg.sort_values('textblob_polarity')
    
    # Create figure
    fig = go.Figure()
    
    # Add TextBlob sentiment bar chart
    fig.add_trace(
        go.Bar(
            y=term_avg['term'],
            x=term_avg['textblob_polarity'],
            name="TextBlob Polarity",
            orientation='h'
        )
    )
    
    # Add vertical line at x=0
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.5,
        x1=0,
        y1=len(term_avg) - 0.5,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Average Sentiment by Mental Health Term",
        xaxis_title="Average Sentiment Polarity (TextBlob)",
        yaxis_title="Mental Health Term",
        height=600
    )
    
    return fig

def create_mental_health_percentage_figure(users_df):
    """
    Create a figure showing mental health tweet percentage distribution
    
    Args:
        users_df (pd.DataFrame): DataFrame containing user metrics
        
    Returns:
        plotly.graph_objects.Figure: Mental health percentage figure
    """
    if 'mental_health_tweet_count' not in users_df.columns or 'tweet_count' not in users_df.columns:
        # Create a dummy figure if data is not available
        fig = go.Figure()
        fig.add_annotation(
            text="User metrics data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Mental Health Tweet Percentage (Data Not Available)",
            height=500
        )
        return fig
    
    # Calculate percentage
    users_df['mental_health_percentage'] = (users_df['mental_health_tweet_count'] / 
                                          users_df['tweet_count']) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=users_df['mental_health_percentage'],
            nbinsx=20
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Mental Health Tweet Percentage by User",
        xaxis_title="Percentage of Tweets Related to Mental Health",
        yaxis_title="Number of Users",
        height=500
    )
    
    return fig

def create_engagement_sentiment_figure(users_df):
    """
    Create a figure showing engagement vs sentiment
    
    Args:
        users_df (pd.DataFrame): DataFrame containing user metrics
        
    Returns:
        plotly.graph_objects.Figure: Engagement vs sentiment figure
    """
    if 'avg_engagement' not in users_df.columns or 'avg_textblob_polarity' not in users_df.columns:
        # Create a dummy figure if data is not available
        fig = go.Figure()
        fig.add_annotation(
            text="User metrics data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(
            title="Engagement vs Sentiment (Data Not Available)",
            height=500
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=users_df['avg_engagement'],
            y=users_df['avg_textblob_polarity'],
            mode='markers',
            marker=dict(
                size=10,
                color=users_df['mental_health_tweet_count'],
                colorscale='Viridis',
                colorbar=dict(title="Mental Health Tweets"),
                showscale=True
            )
        )
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=users_df['avg_engagement'].min(),
        y0=0,
        x1=users_df['avg_engagement'].max(),
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Average Engagement vs Average Sentiment",
        xaxis_title="Average Engagement",
        yaxis_title="Average Sentiment Polarity (TextBlob)",
        height=500
    )
    
    return fig

def generate_static_dashboard():
    """
    Generate static dashboard files for offline viewing
    """
    # Load data
    try:
        tweets_df, terms_df, users_df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create output directory
    static_dir = os.path.join(DASHBOARD_DIR, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    # Generate individual plots
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='textblob_category', data=tweets_df)
    plt.title('TextBlob Sentiment Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='vader_category', data=tweets_df)
    plt.title('VADER Sentiment Distribution')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'sentiment_distribution.png'))
    plt.close()
    
    # 2. Top Mental Health Terms
    plt.figure(figsize=(10, 8))
    top_terms = terms_df.head(15)
    sns.barplot(y='term_text', x='occurrence_count', data=top_terms)
    plt.title('Top 15 Mental Health Terms')
    plt.xlabel('Occurrence Count')
    plt.ylabel('Term')
    plt.tight_layout()
    plt.savefig(os.path.join(static_dir, 'top_terms.png'))
    plt.close()
    
    # 3. User Metrics
    if 'mental_health_tweet_count' in users_df.columns and 'tweet_count' in users_df.columns:
        users_df['mental_health_percentage'] = (users_df['mental_health_tweet_count'] / 
                                              users_df['tweet_count']) * 100
        
        plt.figure(figsize=(10, 6))
        sns.histplot(users_df['mental_health_percentage'], bins=20)
        plt.title('Distribution of Mental Health Tweet Percentage by User')
        plt.xlabel('Percentage of Tweets Related to Mental Health')
        plt.ylabel('Number of Users')
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, 'mental_health_percentage.png'))
        plt.close()
    
    # 4. Engagement vs Sentiment
    if 'avg_engagement' in users_df.columns and 'avg_textblob_polarity' in users_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='avg_engagement', y='avg_textblob_polarity', data=users_df)
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title('Average Engagement vs Average Sentiment')
        plt.xlabel('Average Engagement')
        plt.ylabel('Average Sentiment Polarity (TextBlob)')
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, 'engagement_vs_sentiment.png'))
        plt.close()
    
    # Create HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Social Media and Mental Health Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
            }}
            .row {{
                display: flex;
                flex-wrap: wrap;
                margin: 0 -15px;
            }}
            .col {{
                flex: 1;
                padding: 0 15px;
                min-width: 300px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin-top: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            footer {{
                margin-top: 30px;
                text-align: center;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Social Media and Mental Health Analysis Dashboard</h1>
            
            <div class="section">
                <h2>Overview</h2>
                <p>Total tweets analyzed: {len(tweets_df)}</p>
                <p>Tweets containing mental health terms: {tweets_df['contains_mental_health_term'].sum()}</p>
                <p>Unique mental health terms identified: {len(terms_df)}</p>
                <p>Users analyzed: {len(users_df)}</p>
            </div>
            
            <div class="section">
                <h2>Sentiment Analysis</h2>
                <h3>Sentiment Distribution</h3>
                <img src="static/sentiment_distribution.png" alt="Sentiment Distribution">
            </div>
            
            <div class="section">
                <h2>Mental Health Terms Analysis</h2>
                <h3>Top Mental Health Terms</h3>
                <img src="static/top_terms.png" alt="Top Mental Health Terms">
            </div>
            
            <div class="section">
                <h2>User Analysis</h2>
                <div class="row">
                    <div class="col">
                        <h3>Mental Health Tweet Percentage</h3>
                        <img src="static/mental_health_percentage.png" alt="Mental Health Tweet Percentage">
                    </div>
                    <div class="col">
                        <h3>Engagement vs Sentiment</h3>
                        <img src="static/engagement_vs_sentiment.png" alt="Engagement vs Sentiment">
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Sample Tweets</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Tweet Text</th>
                            <th>Sentiment</th>
                            <th>Mental Health Terms</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    # Add sample tweets
    sample_tweets = tweets_df[tweets_df['contains_mental_health_term'] == 1].head(10)
    for _, row in sample_tweets.iterrows():
        terms = row['extracted_terms']
        terms_str = ', '.join(terms) if isinstance(terms, list) else ''
        
        html_content += f"""
                        <tr>
                            <td>{row['cleaned_text']}</td>
                            <td>{row['textblob_category']} ({row['textblob_polarity']:.2f})</td>
                            <td>{terms_str}</td>
                        </tr>
        """
    
    html_content += f"""
                    </tbody>
                </table>
            </div>
            
            <footer>
                <p>Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Social Media and Mental Health Analysis Project</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_path = os.path.join(DASHBOARD_DIR, 'static_dashboard.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Static dashboard saved to {html_path}")

def main():
    """Main function to create dashboard"""
    print(f"Starting dashboard generation at {datetime.now().isoformat()}")
    
    try:
        # Generate static dashboard
        generate_static_dashboard()
        
        # Create interactive dashboard
        app = create_dashboard()
        
        print(f"Dashboard generation completed at {datetime.now().isoformat()}")
        print(f"Dashboard files saved to {DASHBOARD_DIR}")
        
        # Return app for running if needed
        return app
        
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        return None

if __name__ == "__main__":
    main()
