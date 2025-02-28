import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go  # Add this import
from ui_helper import inject_custom_css, card, stat_card, badge, section_header
from ml_helper import AutoML
from web_data_helper import WebDataIntegration

# Must be the first Streamlit command
st.set_page_config(
    page_title="CSV Data Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# After page config, inject custom CSS and other styles
inject_custom_css()

# Custom sidebar styling
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child { width: 350px; }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child { width: 350px; margin-left: -350px; }
    </style>
""", unsafe_allow_html=True)

# Main header with enhanced styling
st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f2937; font-size: 2.5rem; font-weight: bold;">CSV Data Visualizer</h1>
        <p style="color: #6b7280; font-size: 1.25rem;">Upload your CSV file and discover insights instantly</p>
    </div>
""", unsafe_allow_html=True)

# File upload with enhanced UI
uploaded_file = st.file_uploader("", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Move column type detection to the beginning
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        # Enhanced Data Overview with more insights
        section_header("📊 Dataset Overview", "Comprehensive analysis of your data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stat_card("Total Records", f"{len(df):,}")
        with col2:
            stat_card("Total Features", f"{len(df.columns):,}")
        with col3:
            stat_card("Data Types", f"{len(df.dtypes.unique())} types")
        with col4:
            stat_card("Memory Usage", f"{df.memory_usage().sum() / 1024**2:.1f} MB")

        # Enhanced Data Profile
        st.subheader("📋 Data Profile")
        profile_tabs = st.tabs(["Sample Data", "Column Info", "Quick Stats"])
        
        with profile_tabs[0]:
            st.dataframe(df.head(10), use_container_width=True)
            
        with profile_tabs[1]:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique Values': df.nunique(),
                'Sample Values': [str(df[col].dropna().sample(3).tolist())[:50] for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
            
        with profile_tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numeric Columns", len(numeric_cols))
                st.metric("Categorical Columns", len(categorical_cols))
            with col2:
                st.metric("Missing Cells", df.isnull().sum().sum())
                st.metric("Duplicate Rows", df.duplicated().sum())

        # Enhanced Statistical Analysis
        st.subheader("📈 Statistical Analysis")
        if numeric_cols:
            stats_tabs = st.tabs(["Summary Statistics", "Distribution Analysis", "Correlations"])
            
            with stats_tabs[0]:
                st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)
            
            with stats_tabs[1]:
                selected_col = st.selectbox("Select column for distribution", numeric_cols)
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x=selected_col, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(df, y=selected_col)
                    st.plotly_chart(fig, use_container_width=True)
            
            with stats_tabs[2]:
                if len(numeric_cols) > 1:
                    fig = px.imshow(df[numeric_cols].corr(), text_auto=True)
                    st.plotly_chart(fig, use_container_width=True)

        # Add Machine Learning Section
        section_header("🤖 Machine Learning Analysis", "Automated ML insights and predictions")
        
        if len(numeric_cols) > 0:
            ml_tabs = st.tabs(["Prediction Model", "Feature Importance", "Model Performance"])
            
            with ml_tabs[0]:
                target_col = st.selectbox("Select Target Variable", df.columns)
                if st.button("Train Model"):
                    with st.spinner("Training model..."):
                        auto_ml = AutoML(df)
                        model_scores = auto_ml.train_model(target_col)
                        st.success("Model trained successfully!")
                        for model, score in model_scores.items():
                            st.metric(model, f"{abs(score):.4f}")
            
            with ml_tabs[1]:
                if 'auto_ml' in locals():
                    importance = auto_ml.get_feature_importance()
                    if importance:
                        fig = px.bar(
                            x=list(importance.keys()),
                            y=list(importance.values()),
                            title="Feature Importance"
                        )
                        st.plotly_chart(fig)
            
            with ml_tabs[2]:
                if 'auto_ml' in locals():
                    sample_data = df.sample(min(5, len(df)))
                    predictions = auto_ml.predict(sample_data)
                    results = pd.DataFrame({
                        'Actual': sample_data[target_col],
                        'Predicted': predictions
                    })
                    st.write("Sample Predictions")
                    st.dataframe(results)

        # Add News Integration Section
        section_header("📰 Related News & Analysis", "External data integration")
        
        web_data = WebDataIntegration()
        
        news_tabs = st.tabs(["Industry News", "Market Analysis", "Similar Patterns"])
        
        with news_tabs[0]:
            keywords = " ".join(df.columns)
            news_df = web_data.get_market_news(keywords)
            if not news_df.empty:
                for _, article in news_df.head(5).iterrows():
                    st.markdown(f"""
                        <div class="card">
                            <h3>{article['title']}</h3>
                            <p>{article['description']}</p>
                            <a href="{article['url']}" target="_blank">Read more</a>
                        </div>
                    """, unsafe_allow_html=True)
        
        with news_tabs[1]:
            industry_type = st.selectbox(
                "Select Industry",
                ["retail", "technology", "finance", "healthcare"]
            )
            comparisons = web_data.compare_with_industry_data(df, industry_type)
            if comparisons:
                for metric, data in comparisons.items():
                    st.metric(
                        metric,
                        f"{data['current']:.2f}",
                        f"{data['difference']:.2f} vs benchmark"
                    )
        
        with news_tabs[2]:
            if numeric_cols:
                pattern_col = st.selectbox("Select column for pattern analysis", numeric_cols)
                patterns = web_data.find_similar_patterns(df, pattern_col)
                if patterns:
                    fig = go.Figure()
                    for idx, score in zip(patterns['similar_indices'], patterns['similarity_scores']):
                        fig.add_trace(go.Scatter(
                            y=patterns['patterns'][idx],
                            name=f"Pattern {idx} (similarity: {score:.2f})"
                        ))
                    st.plotly_chart(fig)

        # Display basic information
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Sample")
            st.dataframe(df.head())
        
        with col2:
            st.subheader("Data Summary")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write("Column Data Types:")
            st.write(df.dtypes)
        
        # Data Statistics
        st.header("Data Statistics")
        
        # Check if there are numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        
        if numeric_cols:
            st.subheader("Numeric Data Summary")
            st.dataframe(df[numeric_cols].describe())
        
        if categorical_cols:
            st.subheader("Categorical Data Summary")
            for col in categorical_cols:
                if df[col].nunique() < 10:  # Only show for columns with reasonable number of categories
                    st.write(f"**{col}** value counts:")
                    st.write(df[col].value_counts())
        
        # Missing values
        st.header("Missing Values Analysis")
        missing_data = df.isnull().sum()
        
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
            st.dataframe(missing_df)
            
            # Visualize missing values
            if not missing_df.empty:
                st.subheader("Missing Values Heatmap")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[missing_df['Column']].isnull(), cmap='viridis', yticklabels=False, cbar=False)
                st.pyplot(fig)
        else:
            st.write("No missing values found in the dataset!")
        
        # Visualizations
        st.header("Data Visualizations")
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Distribution Plots", "Correlation Analysis", "Category Comparison", "Time Series (if applicable)"]
        )
        
        # Add enhanced section headers for each visualization
        if viz_type == "Distribution Plots":
            section_header("📈 Distribution Analysis", "Understand the spread of your numeric data")
            
            if numeric_cols:
                selected_col = st.selectbox("Select column", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(df, x=selected_col, marginal="box", title=f"Histogram of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No numeric columns found for distribution analysis.")
        
        elif viz_type == "Correlation Analysis":
            section_header("🔗 Correlation Analysis", "Discover relationships between variables")
            
            if len(numeric_cols) > 1:
                st.subheader("Correlation Between Numeric Variables")
                
                # Correlation matrix
                corr = df[numeric_cols].corr()
                
                # Heatmap
                fig = px.imshow(corr, text_auto=True, aspect="auto", 
                                title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot for selected variables
                st.subheader("Scatter Plot")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Select X-axis", numeric_cols)
                with col2:
                    y_var = st.selectbox("Select Y-axis", [col for col in numeric_cols if col != x_var or len(numeric_cols) == 1])
                
                color_var = None
                if categorical_cols:
                    color_var = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                    if color_var == "None":
                        color_var = None
                
                fig = px.scatter(df, x=x_var, y=y_var, color=color_var, 
                                title=f"Scatter Plot: {x_var} vs {y_var}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Need at least two numeric columns for correlation analysis.")
        
        elif viz_type == "Category Comparison":
            if categorical_cols and numeric_cols:
                st.subheader("Compare Numeric Values Across Categories")
                
                col1, col2 = st.columns(2)
                with col1:
                    cat_var = st.selectbox("Select Categorical Variable", categorical_cols)
                with col2:
                    num_var = st.selectbox("Select Numeric Variable", numeric_cols)
                
                # Limit to top categories if there are too many
                value_counts = df[cat_var].value_counts()
                if len(value_counts) > 10:
                    st.info(f"Showing only top 10 categories out of {len(value_counts)} for clarity.")
                    top_cats = value_counts.nlargest(10).index
                    plot_df = df[df[cat_var].isin(top_cats)]
                else:
                    plot_df = df
                
                # Bar chart
                fig = px.bar(
                    plot_df.groupby(cat_var)[num_var].mean().reset_index(),
                    x=cat_var, y=num_var,
                    title=f"Average {num_var} by {cat_var}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plot
                fig = px.box(plot_df, x=cat_var, y=num_var, 
                            title=f"Distribution of {num_var} by {cat_var}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Need both categorical and numeric columns for category comparison.")
        
        elif viz_type == "Time Series (if applicable)":
            # Check for date columns
            date_cols = []
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass
            
            if date_cols:
                st.subheader("Time Series Analysis")
                date_col = st.selectbox("Select Date Column", date_cols)
                
                # Convert to datetime
                df[date_col] = pd.to_datetime(df[date_col])
                
                # Select variable to plot over time
                if numeric_cols:
                    num_var = st.selectbox("Select Variable to Plot Over Time", numeric_cols)
                    
                    # Create time series dataframe
                    ts_df = df[[date_col, num_var]].dropna()
                    ts_df = ts_df.sort_values(by=date_col)
                    
                    # Time series plot
                    fig = px.line(ts_df, x=date_col, y=num_var, 
                                title=f"{num_var} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Time aggregation
                    st.subheader("Aggregated Time Series")
                    agg_options = ["Day", "Week", "Month", "Year"]
                    agg_period = st.selectbox("Aggregate by", agg_options)
                    
                    agg_map = {
                        "Day": "D",
                        "Week": "W",
                        "Month": "M",
                        "Year": "Y"
                    }
                    
                    # Create aggregated time series
                    ts_df.set_index(date_col, inplace=True)
                    agg_df = ts_df.resample(agg_map[agg_period]).mean()
                    
                    fig = px.line(agg_df, title=f"{num_var} Aggregated by {agg_period}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Need numeric columns for time series analysis.")
            else:
                st.write("No date columns detected for time series analysis.")
                st.write("Tip: If you have a date column, ensure it's in a recognizable date format.")
    
        # Add footer with enhanced styling
        st.markdown("""
            <div style="text-align: center; margin-top: 3rem; padding: 1rem; background-color: #f8fafc;">
                <p style="color: #6b7280;">CSV Data Visualizer | Built with ❤️ using Streamlit</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing the file: {e}")

else:
    # Enhanced welcome screen
    st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem;">
            <img src="https://img.icons8.com/clouds/200/000000/upload.png" style="margin-bottom: 2rem;"/>
            <h2 style="color: #1f2937; margin-bottom: 1rem;">Upload your CSV file to begin</h2>
            <p style="color: #6b7280;">Drag and drop your file here or click to browse</p>
        </div>
    """, unsafe_allow_html=True)

    # Feature showcase with enhanced styling
    section_header("✨ Features", "What you can do with this app")
    
    features = [
        ("📊 Instant Overview", "Get quick insights about your dataset structure"),
        ("📈 Visual Analytics", "Create beautiful visualizations with a few clicks"),
        ("🔍 Deep Analysis", "Uncover patterns and relationships in your data"),
        ("📱 Responsive Design", "Works perfectly on any device")
    ]
    
    cols = st.columns(2)
    for i, (title, desc) in enumerate(features):
        with cols[i % 2]:
            card(title, desc)

# Add footer
st.markdown("---")
st.markdown("CSV Data Visualizer | Built with Streamlit")
