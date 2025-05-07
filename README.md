# Social Media and Mental Health Analysis Project

## Project Overview
This project analyzes the relationship between social media usage and mental health indicators using data collected from Twitter. The analysis focuses on sentiment patterns, mental health term usage, and user engagement metrics to provide insights into how social media discussions reflect and potentially impact mental health.

## Project Structure

### Data Collection
- `twitter_data_collector.py`: Collects tweets related to mental health topics using the Twitter API
- `data/`: Directory containing raw collected data

### Data Processing
- `sentiment_analyzer.py`: Processes tweets and performs sentiment analysis using TextBlob and VADER
- `data/processed/`: Directory containing processed data files

### Database
- `database_setup.py`: Creates SQLite database and imports processed data
- `data/social_media_mental_health.db`: SQLite database with tables for tweets, sentiment, and user metrics

### Analysis
- `data_analyzer.py`: Performs analysis on the collected data and generates visualizations
- `data/results/`: Directory containing analysis results and visualizations

### Dashboard
- `dashboard_generator.py`: Creates interactive and static dashboards for data visualization
- `data/results/dashboard/`: Directory containing dashboard files

## How to Run the Project

### 1. Collect Data
```bash
python twitter_data_collector.py
```
This script collects tweets related to mental health topics using the Twitter API and saves them to the `data/` directory.

### 2. Process Data
```bash
python sentiment_analyzer.py
```
This script processes the collected tweets, performs sentiment analysis, and saves the processed data to the `data/processed/` directory.

### 3. Set Up Database
```bash
python database_setup.py
```
This script creates the SQLite database and imports the processed data.

### 4. Analyze Data
```bash
python data_analyzer.py
```
This script performs analysis on the data and generates visualizations and a summary report in the `data/results/` directory.

### 5. Generate Dashboard
```bash
python dashboard_generator.py
```
This script creates interactive and static dashboards for visualizing the analysis results.

## Key Findings

The analysis of social media content related to mental health reveals several important insights:

1. **Sentiment Distribution**: Mental health discussions on social media show varied sentiment patterns, with both positive and negative expressions present. This reflects the complex nature of mental health conversations.

2. **Mental Health Terms**: The most frequently mentioned mental health terms include "anxiety," "depression," and "stress," indicating these are prominent topics in social media discussions.

3. **Sentiment by Term**: Different mental health terms are associated with different sentiment patterns. Terms like "therapy" and "self-care" tend to appear in more positive contexts, while terms like "anxiety" and "depression" often appear in more negative contexts.

4. **User Engagement**: There appears to be a relationship between engagement metrics and sentiment in mental health discussions, suggesting that emotional content may drive more interaction.

5. **Content Patterns**: Users who frequently discuss mental health topics show distinct posting patterns compared to the general user base.

## Limitations and Future Work

- **Data Limitations**: The analysis is based on a limited sample of tweets and may not represent all social media platforms or user demographics.
- **Sentiment Analysis Accuracy**: Automated sentiment analysis has inherent limitations in detecting nuance, sarcasm, and context-specific meanings.
- **Causality**: This analysis identifies correlations but cannot establish causal relationships between social media usage and mental health outcomes.

Future work could include:
- Expanding data collection to multiple social media platforms
- Incorporating longitudinal analysis to track changes over time
- Adding demographic analysis to understand differences across user groups
- Developing more sophisticated natural language processing techniques for mental health content analysis

## Conclusion

This project demonstrates the potential of data analysis techniques to provide insights into the relationship between social media and mental health. By analyzing sentiment patterns, term usage, and user engagement, we can better understand how mental health is discussed and potentially influenced by social media platforms. These insights could inform strategies for promoting positive mental health discussions and identifying potential risks in online environments.
