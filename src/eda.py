import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

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
        'math score': 'math_score',
        'reading score': 'reading_score',
        'writing score': 'writing_score'
    })

    # Add total_score if missing
    if 'total_score' not in df.columns:
        df['total_score'] = df['math_score'] + df['reading_score'] + df['writing_score']
        print("📊 Added column: total_score")

    # Add avg_score if missing
    if 'avg_score' not in df.columns:
        df['avg_score'] = df['total_score'] / 3
        print("📈 Added column: avg_score")

    return df



def perform_eda(df):
    """Performs basic EDA and visualizations."""
    print("\n📊 Descriptive statistics:")
    print(df.describe())

    if 'avg_score' not in df.columns:
        numeric_cols    = df.select_dtypes(include=['number']).columns
        df['avg_score'] = df[numeric_cols].mean(axis=1)

    # Distribution of average scores
    plt.figure(figsize=(8, 5))
    sns.histplot(df['avg_score'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Average Scores')
    plt.xlabel('Average Score')
    plt.ylabel('Frequency')
    plt.show()

    # Gender vs Average Score
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='gender', y='avg_score', data=df)
    plt.title('Average Score Distribution by Gender')
    plt.show()

    # Test preparation vs Average Score
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='test preparation course', y='avg_score', data=df)
    plt.title('Impact of Test Preparation on Average Scores')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df[['math score', 'reading score', 'writing score', 'avg_score']].corr(),
                annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap of Scores')
    plt.show()

    print("\n✅ EDA Completed Successfully!")
