import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import streamlit as st
import os
from PIL import Image

def clean_file_content(raw_content):
    try:
        # Detect encoding
        encoding = chardet.detect(raw_content)['encoding']
        if not encoding:
            encoding = 'utf-8'  # default
            
        try:
            # Try reading with detected encoding
            clean_content = raw_content.decode(encoding)
        except UnicodeDecodeError:
            # Fallback to utf-16 with BOM handling
            clean_content = raw_content.decode('utf-16').replace('\x00', '').replace('\xff', '').strip()
            
        clean_lines = []
        for line in clean_content.split('\n'):
            if line.strip():  # Only keep non-empty lines
                clean_lines.append(line.strip())
                
        clean_csv = '\n'.join(clean_lines)
        return io.StringIO(clean_csv)
    except Exception as e:
        st.error(f"Error cleaning file: {e}")
        return None

def read_file(uploaded_file):
    try:
        # Try reading as Excel first
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(uploaded_file)
        else:
            # For CSV or text files, read as bytes first
            raw_content = uploaded_file.getvalue()
            
            # Try standard read first
            try:
                return pd.read_csv(io.BytesIO(raw_content))
            except:
                # If standard read fails, try cleaning the file
                cleaned_file = clean_file_content(raw_content)
                if cleaned_file:
                    return pd.read_csv(cleaned_file)
                else:
                    return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def perform_analysis(df):
    if df is None or df.empty:
        return {"error": "No data available for analysis"}
    
    analysis = {}
    
    # Basic statistics
    analysis['shape'] = df.shape
    analysis['columns'] = df.columns.tolist()
    analysis['dtypes'] = df.dtypes.astype(str).to_dict()
    analysis['missing_values'] = df.isnull().sum().to_dict()
    analysis['descriptive_stats'] = df.describe(include='all').fillna('').to_dict()
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        analysis['correlation_matrix'] = df[numeric_cols].corr().fillna('').to_dict()
        analysis['skewness'] = df[numeric_cols].skew().to_dict()
        analysis['kurtosis'] = df[numeric_cols].kurtosis().to_dict()
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_stats = {}
        for col in categorical_cols:
            categorical_stats[col] = {
                'unique_values': df[col].nunique(),
                'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else '',
                'frequency': df[col].value_counts().to_dict()
            }
        analysis['categorical_stats'] = categorical_stats
    
    return analysis

def generate_visualizations(df):
    visualizations = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Histograms for numeric columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        visualizations[f'hist_{col}'] = fig
    
    # Boxplots for numeric columns
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots()
        df[numeric_cols].boxplot(ax=ax)
        ax.set_title('Boxplot of Numeric Columns')
        visualizations['boxplot'] = fig
    
    # Correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        visualizations['heatmap'] = fig
    
    # Count plots for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Avoid plots with too many categories
            fig, ax = plt.subplots()
            sns.countplot(y=col, data=df, ax=ax)
            ax.set_title(f'Count of {col}')
            visualizations[f'count_{col}'] = fig
    
    return visualizations

def main():
    st.title("📊 Data Analysis Tool")
    st.markdown("""
    Upload your CSV or Excel file to perform comprehensive data analysis.
    The tool will automatically detect file format and encoding.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'txt'])
    
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        
        with st.spinner("Analyzing data..."):
            # Read the file
            df = read_file(uploaded_file)
            
            if df is not None:
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Perform analysis
                analysis = perform_analysis(df)
                
                # Display basic info
                st.subheader("Basic Information")
                col1, col2 = st.columns(2)
                col1.metric("Number of Rows", df.shape[0])
                col2.metric("Number of Columns", df.shape[1])
                
                st.write("**Columns and Data Types:**")
                st.write(pd.DataFrame(analysis['dtypes'].items(), columns=['Column', 'Data Type']))
                
                # Missing values
                st.subheader("Missing Values")
                missing_df = pd.DataFrame(analysis['missing_values'].items(), columns=['Column', 'Missing Values'])
                st.dataframe(missing_df)
                
                # Descriptive statistics
                st.subheader("Descriptive Statistics")
                st.dataframe(pd.DataFrame(analysis['descriptive_stats']))
                
                # Numeric analysis
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    st.subheader("Numeric Columns Analysis")
                    
                    # Correlation
                    if len(numeric_cols) > 1:
                        st.write("**Correlation Matrix:**")
                        st.dataframe(pd.DataFrame(analysis['correlation_matrix']))
                    
                    # Skewness and Kurtosis
                    skew_kurt = pd.DataFrame({
                        'Skewness': analysis['skewness'],
                        'Kurtosis': analysis['kurtosis']
                    })
                    st.write("**Skewness and Kurtosis:**")
                    st.dataframe(skew_kurt)
                
                # Categorical analysis
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    st.subheader("Categorical Columns Analysis")
                    for col, stats in analysis['categorical_stats'].items():
                        st.write(f"**{col}**")
                        st.write(f"Unique values: {stats['unique_values']}")
                        st.write(f"Most frequent value: {stats['top_value']}")
                
                # Visualizations
                st.subheader("Data Visualizations")
                visualizations = generate_visualizations(df)
                
                for name, fig in visualizations.items():
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("Could not read the uploaded file. Please check the file format and try again.")

if __name__ == '__main__':
    main()