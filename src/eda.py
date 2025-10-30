import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data
import streamlit as st

def clean_data(df):
    """Handles missing values and ensures proper data types."""
    print("\n🧹 Checking for missing values before cleaning:")
    print(df.isnull().sum())

    # Fill numeric columns with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\n✅ Missing values handled successfully!")
    return df


def feature_engineering(df):
    """Adds total_score and avg_score columns if not present."""

    # Rename columns to use underscores for easier access
    df = df.rename(columns={
        'math score': 'math score',
        'reading score': 'reading score',
        'writing score': 'writing score'
    })

    # Add total_score if missing
    if 'total_score' not in df.columns:
        df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
        print("📊 Added column: total_score")

    # Add avg_score if missing
    if 'avg_score' not in df.columns:
        df['avg_score'] = df['total_score'] / 3
        print("📈 Added column: avg_score")

    return df



def perform_eda(df):
    st.subheader("📊 Exploratory Data Analysis")

    # Compute avg_score if not already present
    if 'avg_score' not in df.columns:
        df['avg_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

    # 1️⃣ Distribution of average scores
    st.markdown("### Distribution of Average Scores")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['avg_score'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Average Scores')
    ax.set_xlabel('Average Score')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # 2️⃣ Boxplots for individual subject scores
    st.markdown("### Distribution of Scores by Subject")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df[['math score', 'reading score', 'writing score']], ax=ax)
    ax.set_title('Score Distribution Across Subjects')
    ax.set_ylabel('Score')
    st.pyplot(fig)

    # 3️⃣ Correlation heatmap
    st.markdown("### Correlation Heatmap of Scores")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        df[['math score', 'reading score', 'writing score', 'avg_score']].corr(),
        annot=True, cmap='coolwarm', ax=ax
    )
    ax.set_title('Correlation Heatmap of Scores')
    st.pyplot(fig)

    st.success("✅ EDA complete! You can now proceed to predictions.")