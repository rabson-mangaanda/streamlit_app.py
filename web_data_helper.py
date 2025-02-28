import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime, timedelta
import wikipedia
from config import NEWS_API_KEY
import streamlit as st

class WebDataIntegration:
    def __init__(self):
        """Initialize with MediaStack API"""
        self.news_api_key = NEWS_API_KEY
        self.news_base_url = "http://api.mediastack.com/v1/news"
    
    def get_financial_data(self, symbol, period='1y'):
        """Get financial data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data, "Success"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def get_market_news(self, keywords):
        """Get relevant news articles using MediaStack API"""
        try:
            params = {
                'access_key': self.news_api_key,
                'keywords': keywords,
                'languages': 'en',
                'limit': 10,
                'sort': 'published_desc'
            }
            
            response = requests.get(self.news_base_url, params=params)
            if response.status_code == 200:
                news_data = response.json()
                if 'data' in news_data:
                    return pd.DataFrame(news_data['data'])
            
            st.warning("Could not fetch news data")
            return pd.DataFrame()
            
        except Exception as e:
            st.warning(f"Error fetching news: {str(e)}")
            return pd.DataFrame()
    
    def get_wikipedia_summary(self, topic):
        """Get Wikipedia summary for context"""
        try:
            return wikipedia.summary(topic, sentences=5)
        except:
            return "No Wikipedia information found."
    
    def compare_with_industry_data(self, df, industry_type):
        """Compare data with industry benchmarks"""
        # Add your industry benchmark data sources here
        industry_benchmarks = {
            'retail': {
                'avg_growth': 0.05,
                'profit_margin': 0.15,
                'inventory_turnover': 4
            },
            'technology': {
                'avg_growth': 0.12,
                'profit_margin': 0.20,
                'inventory_turnover': 6
            }
            # Add more industries as needed
        }
        
        if industry_type in industry_benchmarks:
            benchmark = industry_benchmarks[industry_type]
            comparison = {}
            
            # Compare metrics
            for metric, value in benchmark.items():
                if metric in df.columns:
                    current_value = df[metric].mean()
                    comparison[metric] = {
                        'current': current_value,
                        'benchmark': value,
                        'difference': current_value - value,
                        'performance': 'Above' if current_value > value else 'Below'
                    }
            
            return comparison
        return None
    
    def find_similar_patterns(self, df, target_column):
        """Find similar patterns in historical data"""
        try:
            # Convert to normalized time series
            series = df[target_column].values.reshape(-1, 1)
            normalized = (series - series.mean()) / series.std()
            
            # Find patterns using rolling windows
            window_size = min(30, len(normalized) // 3)
            patterns = []
            
            for i in range(len(normalized) - window_size):
                window = normalized[i:i + window_size]
                patterns.append(window.flatten())
            
            if patterns:
                # Calculate similarity matrix
                similarity = cosine_similarity(patterns)
                
                # Find most similar patterns
                similar_idx = np.argsort(similarity[-1])[-5:]
                
                return {
                    'patterns': patterns,
                    'similar_indices': similar_idx,
                    'similarity_scores': similarity[-1][similar_idx]
                }
        except:
            pass
        return None
    
    def get_recommendations(self, df, industry_type=None):
        """Generate recommendations based on data and external sources"""
        recommendations = []
        
        # Basic statistical recommendations
        for column in df.select_dtypes(include=[np.number]).columns:
            mean_val = df[column].mean()
            std_val = df[column].std()
            
            if std_val > mean_val * 2:
                recommendations.append({
                    'type': 'variation',
                    'message': f"High variation detected in {column}. Consider investigating outliers.",
                    'priority': 'high'
                })
        
        # Industry comparison recommendations
        if industry_type:
            comparison = self.compare_with_industry_data(df, industry_type)
            if comparison:
                for metric, data in comparison.items():
                    if data['performance'] == 'Below':
                        recommendations.append({
                            'type': 'benchmark',
                            'message': f"{metric} is below industry average by {abs(data['difference']):.2f}",
                            'priority': 'high'
                        })
        
        # Trend recommendations
        for column in df.select_dtypes(include=[np.number]).columns:
            patterns = self.find_similar_patterns(df, column)
            if patterns and any(score > 0.9 for score in patterns['similarity_scores']):
                recommendations.append({
                    'type': 'pattern',
                    'message': f"Similar patterns detected in {column}. Consider seasonal effects.",
                    'priority': 'medium'
                })
        
        return recommendations

