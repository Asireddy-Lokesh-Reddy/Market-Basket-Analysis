# Online Retail Transaction Clustering

## Overview

This project analyzes online retail shopping baskets to discover similarities between transactions and explore purchasing patterns. It uses data cleaning, descriptive statistics, visualization, binary transaction encoding, Principal Component Analysis (PCA), and four clustering algorithms: K-means, Agglomerative Hierarchical Clustering, Spectral Clustering, and DBSCAN.

The clustering methods are compared using the **Silhouette Score** and **Davies–Bouldin Index**.

## Problem Statement

The online retail store has transaction records containing multiple products purchased together, but it lacks a systematic analysis of transaction characteristics and groups of similar shopping baskets. This project applies statistical and machine learning techniques to examine transaction data, compare clustering approaches, and draw conclusions about similarities among shopping baskets.

## Objectives

- Understand the structure and characteristics of the retail dataset.
- Clean and preprocess transaction records using Pandas.
- Calculate transaction statistics and identify frequently occurring products.
- Visualize product frequencies and transaction-size distributions.
- Convert shopping baskets into numerical feature vectors.
- Apply PCA for dimensionality reduction and visualization.
- Implement K-means, hierarchical clustering, Spectral Clustering, and DBSCAN.
- Compare clustering performance using the Silhouette Score and Davies–Bouldin Index.
- Discuss limitations and possible business applications.

## Project Structure

A suggested repository layout is:

```text
online-retail-transaction-clustering/
├── basket.py
├── basket.ipynb
├── Online retail_new.csv
├── Case Study.docx
└── README.md
```

You can rename files to simpler names, such as `online_retail_clustering.py`, `online_retail_clustering.ipynb`, `online_retail.csv`, and `case_study.docx`. If you rename the CSV, update its filename in the code.

## Technologies Used

- **Python**
- **Pandas** — data manipulation and cleaning
- **NumPy** — numerical operations
- **Matplotlib** — visualization
- **Scikit-learn** — transaction encoding, PCA, clustering, and evaluation
- **Jupyter Notebook** — interactive development and analysis

## Dataset

The project uses a transaction dataset in which each row represents a shopping basket and each column contains a product recorded for that transaction. Empty cells indicate that no additional product is recorded in that position.

The notebook loads the CSV file without a header row:

```python
df = pd.read_csv("Online retail_new.csv", header=None)
```

Ensure the CSV file is in the working directory, or change the path in the code. Before redistributing the dataset, check its original source and license. If redistribution is not permitted, provide instructions for obtaining the data instead of committing it to GitHub.

## Methodology

### 1. Data Loading and Understanding

Load the transaction data into a Pandas DataFrame and inspect sample records, the first and last rows, and the dataset dimensions.

### 2. Data Cleaning

- Remove leading and trailing spaces from product names.
- Replace empty strings with missing values.
- Count missing values and duplicate transaction rows.
- Remove completely empty rows.
- Review duplicates carefully, because identical baskets may represent legitimate repeated purchases.

### 3. Statistical Analysis

Calculate the total number of transactions and descriptive statistics for transaction size, including count, mean, standard deviation, minimum, quartiles, and maximum. Count product occurrences to identify the ten most frequent products.

### 4. Data Visualization

Generate:
- A bar chart of the ten most frequently occurring products.
- A histogram showing the distribution of transaction sizes.
- A two-dimensional scatter plot of transactions after PCA.

These charts summarize the data. Product frequency alone does not show which products are commonly purchased together.

### 5. Transaction Encoding

Use `MultiLabelBinarizer` to convert each shopping basket into a binary feature vector. Each unique product becomes a feature:

- `1` means the product appears in the transaction.
- `0` means the product does not appear in the transaction.

### 6. Principal Component Analysis (PCA)

Apply PCA to reduce the encoded transaction features to two principal components. The resulting two-dimensional representation is used for visualization and as the input to the clustering algorithms in this implementation.

Dimensionality reduction can discard information, so clusters in the PCA space may not fully reflect similarities in the original feature space.

### 7. Clustering Algorithms

The project implements four methods:

| Algorithm | Description |
|---|---|
| K-means | Partitions transactions into a specified number of clusters using distances to cluster centroids. |
| Agglomerative Hierarchical Clustering | Builds groups by progressively merging similar observations. |
| Spectral Clustering | Uses a nearest-neighbor affinity graph to identify groups. |
| DBSCAN | Groups dense regions and can label isolated observations as noise. It does not require a preset number of clusters. |

In the current implementation, K-means, Agglomerative Clustering, and Spectral Clustering request three clusters. DBSCAN uses `eps=0.1` and `min_samples=5`.

### 8. Evaluation Metrics

The clustering algorithms are compared using:

- **Silhouette Score:** Ranges from -1 to +1; higher values generally indicate better-separated clusters.
- **Davies–Bouldin Index:** Measures similarity between clusters based on compactness and separation; lower values generally indicate better clustering.

The current code calculates these metrics using the two PCA components. The scores therefore describe clustering in the reduced space, not directly in the original product-feature space. DBSCAN may produce noise points or an invalid number of clusters for these metrics, so its labels should be checked before interpreting or calculating scores.

## Installation

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter openpyxl
```

`openpyxl` is needed only if you read or write Excel files as part of your workflow.

## How to Run

1. Clone or download the repository.
2. Ensure the CSV file is in the expected working directory and named `Online retail_new.csv`, or update the filename in the code.
3. Open a terminal in the project folder.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `basket.ipynb` and run the cells in order.

To run the Python script instead:

```bash
python basket.py
```

Make sure the CSV path is correct and all required packages are installed.

## Results

Running the notebook produces transaction statistics, the ten most frequent products, visualizations, the PCA representation, cluster labels, and clustering evaluation scores.

**Add your actual results here after running the notebook.** Record the Silhouette Score and Davies–Bouldin Index for each algorithm where the metrics are valid. Do not claim that one algorithm is best until you have compared the measured results.

## Limitations

- **Information loss:** Reducing a high-dimensional product matrix to two PCA components can discard useful information.
- **Parameter dependence:** Results depend on algorithm settings, including the number of clusters, neighborhood size, and DBSCAN parameters.
- **Duplicate transactions:** Repeated baskets may be genuine purchases, but they can influence clustering.
- **Limited context:** Transaction data alone does not explain why customers purchase particular products.
- **Evaluation space:** Metrics calculated in the PCA space may not fully represent cluster quality in the original feature space.
- **Metric validity:** Silhouette and Davies–Bouldin scores require suitable cluster labels; DBSCAN results need checks for noise and the number of distinct clusters.

## Possible Business Applications

- Explore groups of similar shopping baskets.
- Investigate common transaction patterns.
- Support further analysis of product placement, promotions, or customer shopping behavior.

These are potential applications; the clustering results alone do not establish that a business intervention will be effective.

## Future Improvements

- Compare clustering quality in both PCA and original feature spaces where computationally practical.
- Tune algorithm parameters systematically.
- Handle DBSCAN noise labels and invalid metric cases explicitly.
- Explore alternative dimensionality-reduction techniques or transaction similarity measures.
- Use association-rule mining to investigate products that occur together, since frequency counts alone do not identify product combinations.


