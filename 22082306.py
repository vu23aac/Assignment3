# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 19:59:06 2024

@author: prasanna_udara
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import cluster_tools as ct
import errors as err

def read_data(filename):
    '''
    Function to read the dataframe "filename" and return the two dataframes
    having the required indicator in the dataframe. Returns dataframe and
    transpose of it.
    '''

    # Selected Indicator list
    indicator_lst = [
        "Energy use (kg of oil equivalent per capita)",
        "CO2 emissions (metric tons per capita)"]

    # Read the csv file as dataframe and skip 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # Select the desired indicators and country names from the original data
    df = df[df["Indicator Name"].isin(indicator_lst)]

    # Drop the country code, Indicator code and Unnamed: 67
    df.drop(["Unnamed: 67", "Country Code", "Indicator Code"], axis=1,
            inplace=True)

    # Return the dataframes
    return df, df.transpose()


def get_cluster_num(data):
    """
    Function defined to calculate the optimal cluster number for the given data
    and return the best cluster number for clustering.
    """
    # Define the list of clusters and scores
    clusters = []
    scores = []

    # loop over number of clusters
    for ncluster in range(2, 10):

        # Setting up clusters over number of clusters
        kmeans = cluster.KMeans(n_clusters=ncluster, n_init=10)

        # Cluster fitting
        kmeans.fit(data)

        # Get the labels
        labels = kmeans.labels_

        # Add the values to the list
        clusters.append(ncluster)
        scores.append(skmet.silhouette_score(data, labels))

    clusters = np.array(clusters)
    scores = np.array(scores)

    # Get the best cluster number
    best_ncluster = clusters[scores == np.max(scores)]

    # Return the best cluster number
    return best_ncluster[0]


def plot_cluster_data(data, ncluster, year):
    """
    Function defined to illustrate cluster plot using dataframe, cluster number
    and year. It takes the data clusters them into the required number of
    optimal clusters and plot it using lineplot with the cluster centers.
    """

    # Using the library get the norm, min and max using scalar
    df_norm, df_min, df_max = ct.scaler(data)

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=10)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)

    # extract cluster labels
    labels = kmeans.labels_

    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    # Scale back the original centres
    cen = ct.backscale(cen, df_min, df_max)

    # Get the locations
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    # extract x and y values of data points
    x = data["CO2 emissions (metric tons per capita)"]
    y = data["Energy use (kg of oil equivalent per capita)"]

    # Intialization of the figure
    plt.figure(figsize=(8, 5))

    # plot data with kmeans cluster number
    scatter_data = plt.scatter(x, y, 10, labels, marker="o", cmap="Dark2")

    # show cluster centres
    plt.scatter(xkmeans, ykmeans, 50, c='k', marker="*")

    # Title text
    title_txt = f'Countries Cluster of CO2 emissions vs Energy use in {year}'
    # Define the plot title
    plt.title(title_txt)

    # Axes labelling
    plt.xlabel("CO2 emissions (metric tons per capita)")
    plt.ylabel("Energy use (kg of oil equivalent per capita)")

    # Display legend
    legend_labels = [f'Cluster {i}' for i in range(kmeans.n_clusters)]
    plt.legend(handles=scatter_data.legend_elements()[0], labels=legend_labels)

    # Display plot
    plt.show()


def logistic(t, n0, g, t0):
    """
    Function defined to illustrate the Calculation of the logistic function
    with scale factor n0 and growth rate g
    """

    # Logistic Function equation
    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


def forecast(data, country, fit_fun=logistic):
    """
    Function defined to illustrate the fitting and forecast of the data and
    make predictions till year 2030.

    """
    # Extract data with country name
    country_data = data[data["Country Name"] == country]

    # Extract data with Indicator Name
    country_data = country_data[country_data["Indicator Name"] ==
                                "Energy use (kg of oil equivalent per capita)"]

    # Drop the Indicator Name
    country_data.drop(["Indicator Name"], axis=1, inplace=True)

    # Drop Nan values from the desired indicator values
    country_data.dropna(axis=1, inplace=True)

    # Set the index with Country Name
    country_data.set_index("Country Name", inplace=True)

    # Create the dataframe having Country Name as column
    country_col_df = country_data.stack().unstack(level=0)

    # Change the datatype of index to int
    country_col_df.index = country_col_df.index.astype(int)

    # Curve fit the data
    param, covar = opt.curve_fit(logistic, country_col_df.index,
                                 country_col_df[country],
                                 p0=(2e10, 0.20, 1980))

    # Create an year range
    year_range = np.arange(1960, 2030)

    # Calculate the sigma value
    sigma = err.error_prop(year_range, logistic, param, covar)

    # Forcast the data using logistic function
    forecast = logistic(year_range, *param)

    # Get the lower and upper limits of the condence range
    low, up = err.err_ranges(year_range, logistic, param, sigma)

    # Intialization of the figure
    plt.figure()

    # Calling the plot function to display Energy Use lineplot
    plt.plot(country_col_df.index, country_col_df[country], label="Energy Use")

    # Calling the plot function to display Forcast lineplot
    plt.plot(year_range, forecast, label="Forecast", color='red')

    # Plot the Confidence Range
    plt.fill_between(year_range, low, up, color="yellow", alpha=0.7,
                     label='Confidence Margin')

    # Define the plot title
    plt.title(f"Energy Use in Kg Oil equivalent forecast for {country}")

    # Axes labelling
    plt.xlabel("Year")
    plt.ylabel("Energy Use in Kg Oil equivalent per capita")

    # Display legend
    plt.legend()

    # Display plot
    plt.show()


def main():
    # Calling read_data function with filename
    df, df_transpose = read_data("API_19_DS2_en_csv_v2_5998250.csv")

    # Defining the data for clustering
    df_year = df[["Country Name", "Indicator Name", "1990"]]
    df_year = df_year.pivot(index='Country Name', columns='Indicator Name',
                            values="1990")
    df_year.dropna(axis=0, inplace=True)

    # Calling get_cluster_num function with data
    n_cluster = get_cluster_num(df_year)

    # Calling plot_cluster_data function with data, cluster number and year
    plot_cluster_data(df_year, n_cluster, 1990)

    # Calling plot_cluster_data function with data and Country
    forecast(df, "Germany")
    forecast(df, "United States")


if __name__ == "__main__":
    # Calling main program
    main()
