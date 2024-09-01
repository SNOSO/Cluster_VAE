# README

## Overview

This project is designed to evaluate the performance of clustering and classification algorithms on spatial transcriptomics data, particularly focusing on data from the Dorsolateral Prefrontal Cortex (DLPFC). The analysis includes data preprocessing, dimensionality reduction using a Variational Autoencoder (VAE), and subsequent classification using a neural network model.

## Requirements

Ensure that you have the following libraries installed before running the code:

- `caret`
- `keras`
- `tensorflow`
- `tfautograph`

You can install these packages in R using the following commands:

```R
install.packages("caret")
install.packages("tensorflow")
library(tensorflow)
install_keras()
```

## Data Preparation

The project uses two key datasets:

1. `RefSeuratObj.RDS`: The reference Seurat object containing spatial transcriptomics data.
2. `DLPFCGeneList.RDS`: A list of selected genes for the analysis.

### Loading Data

The Seurat object and gene list are loaded using the following commands:

```R
DLPFC <- readRDS("RefSeuratObj.RDS")
gl <- readRDS("DLPFCGeneList.RDS")
```

### Subsetting Data

The expression matrix is subsetted to include only the selected genes from the gene list:

```R
DLPFC@assays$SCT@scale.data <- DLPFC@assays$SCT@scale.data[which(rownames(DLPFC@assays$SCT@scale.data) %in% gl),]
```

### Extracting Necessary Components

The matrix, coordinates, and labels are extracted as follows:

```R
matrix <- DLPFC@assays$SCT@scale.data
coordinates <- DLPFC@images$slice1@coordinates[,c(1:3)]
labels <- DLPFC$Labs
```

## Variational Autoencoder (VAE) Model

A VAE model is implemented to perform dimensionality reduction on the spatial transcriptomics data. The encoder and decoder are constructed using custom `keras` layers, with the encoder reducing the dimensionality of the input data and the decoder reconstructing it.

### VAE Implementation

The VAE model is defined and trained using the following code:

```R
original_dim <- nrow(matrix)
vae <- VariationalAutoEncoder(original_dim, 95, 65, 18)

optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = 1e-3)
vae %>% compile(optimizer, loss = loss_mean_squared_error())
vae %>% fit(
  t(matrix), 
  t(matrix), 
  epochs = 100,
  batch_size = 64,
  callbacks = list(callback_early_stopping(monitor = 'loss', patience = 5))
)
```

### Feature Extraction

The trained VAE encoder is used to extract features from the matrix:

```R
extract_features <- function(matrix) {
  encoder <- vae$encoder
  features <- encoder(matrix)
  return(features[[3]])
}
```

## Neighborhood Averaging and Classification

The `RunBEAR` function performs neighborhood averaging on the matrix, trains a classifier on the averaged data, and evaluates the model's performance on a test set.

### Running the Function

```R
RunBEAR(matrix = matrix, coordinates = coordinates, labels = labels, nsize = 3, output_file = "~/spatial_cluster_evaluation/metrics.txt")
```

The function performs the following steps:

1. **Neighborhood Averaging:** Averages the gene expression data for neighboring spots in the spatial transcriptomics dataset.
2. **Training a Classifier:** Trains a neural network classifier using the extracted features from the VAE encoder.
3. **Model Evaluation:** Tests the model on a held-out test set and saves the confusion matrix and classification metrics to a specified output file.

## Output

The function will save the confusion matrix and evaluation metrics to `metrics.txt` in the specified directory.

## Usage

To run the analysis, ensure that the datasets are correctly located at the paths specified, then execute the script in R. The script will handle data loading, processing, model training, and evaluation, outputting the results to the specified file.
