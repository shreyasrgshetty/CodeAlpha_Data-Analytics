import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

class SentimentAnalyzer:
    def _init_(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion lexicon for detecting specific emotions
        self.emotion_lexicon = {
            'joy': ['happy', 'joy', 'excited', 'wonderful', 'amazing', 'great', 'love', 'excellent', 'fantastic'],
            'sadness': ['sad', 'unhappy', 'disappointed', 'terrible', 'awful', 'bad', 'hate', 'poor', 'worst'],
            'anger': ['angry', 'furious', 'annoyed', 'frustrated', 'irritated', 'mad'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected']
        }
    
    def classify_sentiment(self, text):
        """
        Classify text as positive, negative, or neutral
        Uses VADER sentiment analyzer
        """
        if not text or pd.isna(text):
            return 'neutral'
        
        scores = self.sia.polarity_scores(str(text))
        compound = scores['compound']
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def get_sentiment_scores(self, text):
        """Get detailed sentiment scores"""
        if not text or pd.isna(text):
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}
        
        return self.sia.polarity_scores(str(text))
    
    def detect_emotions(self, text):
        """
        Detect specific emotions using NLP and lexicon
        """
        if not text or pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        detected_emotions = []
        
        for emotion, keywords in self.emotion_lexicon.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        return detected_emotions
    
    def analyze_text(self, text):
        """Complete analysis of a single text"""
        sentiment = self.classify_sentiment(text)
        scores = self.get_sentiment_scores(text)
        emotions = self.detect_emotions(text)
        
        return {
            'text': text,
            'sentiment': sentiment,
            'compound_score': scores['compound'],
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu'],
            'emotions': emotions
        }
    
    def analyze_dataset(self, texts, source_name='data'):
        """Analyze a list or series of texts"""
        results = []
        for text in texts:
            analysis = self.analyze_text(text)
            analysis['source'] = source_name
            results.append(analysis)
        
        return pd.DataFrame(results)
    
    def analyze_reviews(self, reviews_df, text_column='review', rating_column=None):
        """
        Analyze Amazon reviews or similar review data
        """
        results = self.analyze_dataset(reviews_df[text_column], source_name='Amazon Reviews')
        
        if rating_column and rating_column in reviews_df.columns:
            results['rating'] = reviews_df[rating_column].values
        
        return results
    
    def analyze_social_media(self, posts_df, text_column='text', platform='social_media'):
        """
        Analyze social media posts
        """
        results = self.analyze_dataset(posts_df[text_column], source_name=platform)
        
        # Add social media specific metrics if available
        if 'likes' in posts_df.columns:
            results['likes'] = posts_df['likes'].values
        if 'shares' in posts_df.columns:
            results['shares'] = posts_df['shares'].values
        
        return results
    
    def get_sentiment_summary(self, df):
        """
        Generate summary statistics of sentiment analysis
        """
        summary = {
            'total_texts': len(df),
            'positive_count': len(df[df['sentiment'] == 'positive']),
            'negative_count': len(df[df['sentiment'] == 'negative']),
            'neutral_count': len(df[df['sentiment'] == 'neutral']),
            'positive_percentage': (len(df[df['sentiment'] == 'positive']) / len(df) * 100),
            'negative_percentage': (len(df[df['sentiment'] == 'negative']) / len(df) * 100),
            'neutral_percentage': (len(df[df['sentiment'] == 'neutral']) / len(df) * 100),
            'avg_compound_score': df['compound_score'].mean(),
            'most_common_emotions': self._get_top_emotions(df)
        }
        return summary
    
    def _get_top_emotions(self, df, top_n=5):
        """Get most common emotions from analysis"""
        all_emotions = []
        for emotions in df['emotions']:
            all_emotions.extend(emotions)
        
        if not all_emotions:
            return []
        
        emotion_counts = Counter(all_emotions)
        return emotion_counts.most_common(top_n)
    
    def identify_trends(self, df, time_column=None):
        """
        Identify sentiment trends over time or patterns
        """
        trends = {
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_scores_by_sentiment': df.groupby('sentiment')['compound_score'].mean().to_dict()
        }
        
        if time_column and time_column in df.columns:
            df_time = df.copy()
            df_time[time_column] = pd.to_datetime(df_time[time_column])
            trends['sentiment_over_time'] = df_time.groupby([time_column, 'sentiment']).size().unstack(fill_value=0)
        
        return trends
    
    def generate_insights(self, df, source_name='data'):
        """
        Generate actionable insights for marketing, product development, or social insights
        """
        summary = self.get_sentiment_summary(df)
        
        insights = {
            'marketing_insights': [],
            'product_insights': [],
            'social_insights': []
        }
        
        # Marketing insights
        if summary['positive_percentage'] > 70:
            insights['marketing_insights'].append(
                f"Strong positive sentiment ({summary['positive_percentage']:.1f}%) - leverage testimonials and reviews in campaigns"
            )
        elif summary['negative_percentage'] > 30:
            insights['marketing_insights'].append(
                f"High negative sentiment ({summary['negative_percentage']:.1f}%) - focus on reputation management"
            )
        
        # Product insights
        negative_texts = df[df['sentiment'] == 'negative']['text'].tolist()
        if negative_texts:
            insights['product_insights'].append(
                "Analyze negative feedback for product improvement opportunities"
            )
        
        common_emotions = summary['most_common_emotions']
        if common_emotions:
            top_emotion = common_emotions[0][0]
            insights['product_insights'].append(
                f"Primary emotion: {top_emotion} - align product messaging accordingly"
            )
        
        # Social insights
        if summary['neutral_percentage'] > 50:
            insights['social_insights'].append(
                "High neutral sentiment - consider more engaging content to drive emotional response"
            )
        
        insights['social_insights'].append(
            f"Sentiment score: {summary['avg_compound_score']:.3f} - {'positive' if summary['avg_compound_score'] > 0 else 'negative'} overall trend"
        )
        
        return insights
    
    def visualize_results(self, df, title='Sentiment Analysis Results'):
        """
        Create visualizations of sentiment analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Sentiment Distribution')
        
        # Compound score distribution
        axes[0, 1].hist(df['compound_score'], bins=30, edgecolor='black')
        axes[0, 1].set_title('Compound Score Distribution')
        axes[0, 1].set_xlabel('Compound Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # Sentiment scores comparison
        score_data = df.groupby('sentiment')[['positive_score', 'negative_score', 'neutral_score']].mean()
        score_data.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Scores by Sentiment')
        axes[1, 0].set_xlabel('Sentiment')
        axes[1, 0].set_ylabel('Average Score')
        axes[1, 0].legend(['Positive', 'Negative', 'Neutral'])
        
        # Top emotions
        all_emotions = []
        for emotions in df['emotions']:
            all_emotions.extend(emotions)
        if all_emotions:
            emotion_counts = Counter(all_emotions).most_common(10)
            emotions, counts = zip(*emotion_counts)
            axes[1, 1].barh(emotions, counts)
            axes[1, 1].set_title('Top Emotions Detected')
            axes[1, 1].set_xlabel('Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No emotions detected', ha='center', va='center')
            axes[1, 1].set_title('Top Emotions Detected')
        
        plt.tight_layout()
        return fig


# Example Usage
if _name_ == "_main_":
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Example 1: Analyze Amazon reviews
    print("=" * 50)
    print("EXAMPLE 1: AMAZON REVIEWS ANALYSIS")
    print("=" * 50)
    
    amazon_reviews = pd.DataFrame({
        'review': [
            "This product is amazing! Best purchase ever.",
            "Terrible quality. Broke after one week.",
            "It's okay, nothing special.",
            "Love it! Exceeded my expectations.",
            "Waste of money. Very disappointed."
        ],
        'rating': [5, 1, 3, 5, 2]
    })
    
    amazon_results = analyzer.analyze_reviews(amazon_reviews)
    print("\nAnalysis Results:")
    print(amazon_results[['text', 'sentiment', 'compound_score', 'emotions']])
    
    print("\nSummary Statistics:")
    summary = analyzer.get_sentiment_summary(amazon_results)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Example 2: Social media analysis
    print("\n" + "=" * 50)
    print("EXAMPLE 2: SOCIAL MEDIA ANALYSIS")
    print("=" * 50)
    
    social_posts = pd.DataFrame({
        'text': [
            "Just got the new iPhone! So excited!",
            "Customer service was awful today.",
            "The weather is nice.",
            "This app keeps crashing. Very frustrating!",
            "Best day ever! Feeling blessed."
        ],
        'likes': [150, 45, 20, 89, 230]
    })
    
    social_results = analyzer.analyze_social_media(social_posts, platform='Twitter')
    print("\nSocial Media Results:")
    print(social_results[['text', 'sentiment', 'compound_score', 'likes']])
    
    # Example 3: Generate insights
    print("\n" + "=" * 50)
    print("EXAMPLE 3: ACTIONABLE INSIGHTS")
    print("=" * 50)
    
    insights = analyzer.generate_insights(amazon_results, source_name='Amazon')
    print("\nMarketing Insights:")
    for insight in insights['marketing_insights']:
        print(f"  • {insight}")
    
    print("\nProduct Insights:")
    for insight in insights['product_insights']:
        print(f"  • {insight}")
    
    print("\nSocial Insights:")
    for insight in insights['social_insights']:
        print(f"  • {insight}")
    
    # Example 4: Visualizations
    print("\n" + "=" * 50)
    print("EXAMPLE 4: VISUALIZATIONS")
    print("=" * 50)
    print("Creating visualizations...")
    
    fig = analyzer.visualize_results(amazon_results, title='Amazon Reviews Sentiment Analysis')
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'sentiment_analysis.png'")
    plt.show()