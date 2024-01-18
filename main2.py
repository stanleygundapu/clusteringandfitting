import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import silhouette_score

def read_clean_transpose_data(file_path):
    """
    Read, clean, and transpose the data.

    Parameters:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV file.
    - cleaned_data (pd.DataFrame): Cleaned data after removing missing values.
    - transposed_data (pd.DataFrame): Transposed data.
    """
    # Read the original data
    original_data = pd.read_csv(file_path)

    # Select relevant columns
    selected_columns = [
        "Forest area (% of land area) [AG.LND.FRST.ZS]" ,
        "Labor force, female (% of total labor force) [SL.TLF.TOTL.FE.ZS]" ,
        "Employers, total (% of total employment) (modeled ILO estimate) [SL.EMP.MPYR.ZS]"
    ]
    df_selected = original_data[selected_columns]

    # Drop rows with missing values and convert "Forest area" to numeric
    df_selected.dropna(inplace=True)
    df_selected["Forest area (% of land area) [AG.LND.FRST.ZS]"] = pd.to_numeric(
        df_selected["Forest area (% of land area) [AG.LND.FRST.ZS]"] ,
        errors='coerce'
    )

    # Normalize and transpose the data
    df_normalized = (df_selected.iloc[: , 1:] - df_selected.iloc[: , 1:].mean()) / df_selected.iloc[: , 1:].std()
    transposed_data = df_normalized.transpose()

    return original_data, df_selected, transposed_data


def kmeans_clustering(data , features , n_clusters=3):
    """
    Perform K-Means clustering.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of feature columns for clustering.
    - n_clusters (int): Number of clusters.

    Returns:
    - clustered_data (pd.DataFrame): Data with additional 'Cluster' column.
    - cluster_centers (np.ndarray): Coordinates of cluster centers.
    """
    kmeans = KMeans(n_clusters=n_clusters , random_state=42)
    clustered_data = data.copy()
    clustered_data['Cluster'] = kmeans.fit_predict(clustered_data[features])
    cluster_centers = kmeans.cluster_centers_

    return clustered_data , cluster_centers


def calculate_silhouette_score(data , features , cluster_column):
    """
    Calculate the silhouette score for K-Means clustering.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - features (list): List of feature columns for clustering.
    - cluster_column (str): Name of the column containing cluster labels.

    Returns:
    - silhouette_avg (float): Silhouette score.
    """
    silhouette_avg = silhouette_score(data[features] , data[cluster_column])
    return silhouette_avg


def plot_kmeans_clusters(data , x_column , y_column , cluster_column , cluster_centers):
    """
    Plot K-Means clusters.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - x_column (str): Name of the x-axis column.
    - y_column (str): Name of the y-axis column.
    - cluster_column (str): Name of the column containing cluster labels.
    - cluster_centers (np.ndarray): Coordinates of cluster centers.
    """
    plt.scatter(data[x_column] , data[y_column] , c=data[cluster_column] , cmap='viridis')
    plt.scatter(cluster_centers[: , 0] , cluster_centers[: , 1] , marker='x' ,
                s=200 , linewidths=3 , color='r')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title('K-Means Clustering')
    plt.show()


# Calculate confidence intervals for predicted values
def confidence_interval(params , covariance , x_values , confidence=0.95):
    """
        Calculate confidence intervals for the predicted values.

        Parameters:
        - params (ndarray): Array of optimized parameters from curve fitting.
        - covariance (ndarray): Covariance matrix obtained from curve fitting.
        - x_values (ndarray): Input values for which predictions and confidence intervals are calculated.
        - confidence (float, optional): Confidence level for the interval. Default is 0.95.

        Returns:
        - lower_bound (ndarray): Lower bounds of the confidence interval for each predicted value.
        - upper_bound (ndarray): Upper bounds of the confidence interval for each predicted value.
        """
    alpha = 1 - confidence
    n = len(x_values)
    dof = max(0 , n - len(params))
    t_value = stats.t.ppf(1 - alpha / 2 , dof)
    error = np.sqrt(np.diag(covariance))

    lower_bound = np.zeros(n)
    upper_bound = np.zeros(n)

    for i in range(min(n , len(params))):
        lower_bound[i] = model_function(x_values[i] , *params) - t_value * error[i]
        upper_bound[i] = model_function(x_values[i] , *params) + t_value * error[i]

    return lower_bound , upper_bound


# Define a simple model function (e.g., polynomial of degree 1)
def model_function(x , a , b):
    """
        Evaluate a linear model for given input values.

        Parameters:
        - x (array-like): Input values.
        - a (float): Coefficient for the linear term.
        - b (float): Intercept term.

        Returns:
        - y (array-like): Output values predicted by the linear model.
        """
    x_array = np.asarray(x , dtype=float)
    return a * x_array + b


# Load, clean, and transpose the data
original_data , cleaned_data , transposed_data = \
    read_clean_transpose_data('ab4b9746-ac40-4b82-b790-33e1520117f8_Data.csv')

# Perform K-Means clustering
features_for_clustering = ["Labor force, female (% of total labor force) [SL.TLF.TOTL.FE.ZS]" ,
                            "Employers, total (% of total employment) (modeled ILO estimate) [SL.EMP.MPYR.ZS]"]
cleaned_data , cluster_centers = kmeans_clustering(cleaned_data , features_for_clustering)

# Calculate the silhouette score
silhouette_avg = calculate_silhouette_score(cleaned_data , features_for_clustering , 'Cluster')
print("Silhouette Score:" , silhouette_avg)

# Plotting the clusters
plot_kmeans_clusters(cleaned_data , features_for_clustering[0] , features_for_clustering[1] ,
                     'Cluster' , cluster_centers)


# Curve fitting and plotting code remains unchanged
years = original_data['Time']
urban_population = original_data['Employers, total (% of total employment) (modeled ILO estimate) [SL.EMP.MPYR.ZS]']

# Remove or replace NaN and infinite values
mask = np.isfinite(urban_population)
urban_population = urban_population[mask]
years = years[mask]

# Convert 'years' to strings
years_str = years.astype(str)

# Convert to NumPy array if using pandas Series
urban_population = urban_population.values
years_str = years_str.values

# Fit the model to the data
params , covariance = curve_fit(model_function , years_str , urban_population)

# Generate predictions for future years
future_years = np.array([2020, 2030, 2040])
predicted_values_future = model_function(future_years , *params)
print(future_years)
print(predicted_values_future)

# Combine actual years and future years
all_years = np.concatenate([years_str , future_years])
predicted_values_all = np.concatenate([urban_population , predicted_values_future])

# Calculate confidence intervals for predicted values
lower_bound , upper_bound = confidence_interval(params , covariance , all_years)

# Plot the results
plt.scatter(years_str , urban_population , label='Actual Data' , color='pink')
plt.plot(all_years.astype(str) , predicted_values_all , label='Predicted Values' ,
         color='orange')
plt.fill_between(all_years.astype(str) , lower_bound , upper_bound , color='black' ,
                 alpha=0.2 , label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Employers, total (% of total employment) ')
plt.title('Curve Fitting')
plt.legend()
plt.show()

