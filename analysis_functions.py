import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_target_correlations(target, features):
    """
    Plots the correlation between the target and each numeric feature in the features DataFrame.

    Parameters:
    target (pd.Series): The target variable.
    features (pd.DataFrame): The DataFrame containing feature variables.
    """
    # Select only numeric columns
    numeric_features = features.select_dtypes(include=['number'])

    # Calculate correlations
    correlations = numeric_features.apply(lambda x: x.corr(target))

    # Create a DataFrame for correlations
    corr_df = correlations.reset_index()
    corr_df.columns = ['Feature', 'Correlation']

    # Sort the DataFrame by correlation value
    corr_df = corr_df.sort_values(by='Correlation', ascending=False)

    # Plot the correlations
    plt.figure(figsize=(12, 8))
    sns.barplot(data=corr_df, x='Correlation', y='Feature', palette='coolwarm')
    plt.title('Correlation between Target and Numeric Features')
    plt.xlabel('Correlation')
    plt.ylabel('Feature')
    plt.show()


# Example usage
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    'feature3': [100, 200, 300, 400, 500],
    'feature4': ['a', 'b', 'c', 'd', 'e'],  # Non-numeric feature
    'target': [10, 12, 14, 16, 18]
}

df = pd.DataFrame(data)
target = df['target']
features = df.drop(columns=['target'])

plot_feature_target_correlations(target, features)
