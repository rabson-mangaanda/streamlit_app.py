import pandas as pd
import numpy as np
from scipy import stats

def generate_basic_insights(df):
    insights = []
    
    # Dataset Overview
    insights.append({
        'category': 'Overview',
        'insight': f"Dataset contains {df.shape[0]} records with {df.shape[1]} features",
        'importance': 'high'
    })
    
    # Data Quality
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    if missing_pct.max() > 0:
        insights.append({
            'category': 'Data Quality',
            'insight': f"Data completeness issues found: {missing_pct[missing_pct > 0].index.tolist()}",
            'importance': 'high'
        })
    
    # Numeric Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Distribution Analysis
        skew = df[col].skew()
        if abs(skew) > 1:
            insights.append({
                'category': 'Distribution',
                'insight': f"{col} shows {'positive' if skew > 0 else 'negative'} skew ({skew:.2f}), suggesting unusual distribution",
                'importance': 'medium'
            })
        
        # Outlier Detection
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers_pct = (z_scores > 3).mean() * 100
        if outliers_pct > 5:
            insights.append({
                'category': 'Outliers',
                'insight': f"{col} has {outliers_pct:.1f}% potential outliers that need investigation",
                'importance': 'high'
            })
    
    # Correlation Analysis
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.7)
        high_corr = [(corr_matrix.index[x], corr_matrix.columns[y], corr_matrix.iloc[x, y]) 
                     for x, y in zip(*high_corr) if x != y and x < y]
        
        for var1, var2, corr in high_corr:
            insights.append({
                'category': 'Correlation',
                'insight': f"Strong {'positive' if corr > 0 else 'negative'} correlation ({corr:.2f}) between {var1} and {var2}",
                'importance': 'medium'
            })
    
    return insights

def generate_recommendations(df):
    recommendations = []
    
    # Data Quality Recommendations
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        recommendations.append({
            'category': 'Data Quality',
            'recommendation': f"Consider handling missing values in: {', '.join(missing_cols)}",
            'priority': 'high'
        })
    
    # Feature Engineering Recommendations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) > 0:
        skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 1]
        if skewed_cols:
            recommendations.append({
                'category': 'Feature Engineering',
                'recommendation': f"Consider log transformation for skewed features: {', '.join(skewed_cols)}",
                'priority': 'medium'
            })
    
    # Categorical Encoding Recommendations
    if len(categorical_cols) > 0:
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > 10]
        if high_cardinality_cols:
            recommendations.append({
                'category': 'Feature Engineering',
                'recommendation': f"Consider encoding or grouping categories for: {', '.join(high_cardinality_cols)}",
                'priority': 'medium'
            })
    
    return recommendations
