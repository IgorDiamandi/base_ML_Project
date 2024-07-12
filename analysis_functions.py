import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

def plot_feature_target_correlations(target, features):
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

def calculate_vif(df, column):
    features = df.drop(columns=[column])
    vif_data  = pd.DataFrame()
    vif_data['Feature'] = features.columns
    vif_data['VIF'] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    return vif_data