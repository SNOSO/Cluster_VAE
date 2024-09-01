library(caret)
library(keras)
library(tensorflow)
library(tfautograph)

# Load data

DLPFC <- readRDS("RefSeuratObj.RDS")
gl <- readRDS("DLPFCGeneList.RDS")

#subset
DLPFC@assays$SCT@scale.data <-
  DLPFC@assays$SCT@scale.data[which(rownames(DLPFC@assays$SCT@scale.data) %in% 
                                      gl),] #from 3000 genes to 185 genes

matrix <- DLPFC@assays$SCT@scale.data

coordinates <- DLPFC@images$slice1@coordinates[,c(1:3)]

labels <- DLPFC$Labs


# VAE model
Sampling(keras$layers$Layer) %py_class% {
  call <- function(inputs) {
    c(z_mean, z_log_var) %<-% inputs
    batch <- tf$shape(z_mean)[1]
    dim <- tf$shape(z_mean)[2]
    epsilon <- k_random_normal(shape = c(batch, dim))
    z_mean + exp(0.5 * z_log_var) * epsilon
  }
}

Encoder(keras$layers$Layer) %py_class% {
  initialize <- function(latent_dim = 18, intermediate_dim = 95, inter_dim2 = 65, name = "encoder", ...) {
    super$initialize(name = name, ...)
    self$dense_proj <- layer_dense(units = intermediate_dim, activation = "relu")
    self$dense_proj2 <- layer_dense(units = inter_dim2, activation = "relu")
    self$dense_mean <- layer_dense(units = latent_dim)
    self$dense_log_var <- layer_dense(units = latent_dim)
    self$sampling <- Sampling()
  }
  
  call <- function(inputs) {
    x <- self$dense_proj(inputs)
    x <- self$dense_proj2(x)  # Ensure the flow through the second layer
    z_mean <- self$dense_mean(x)
    z_log_var <- self$dense_log_var(x)
    z <- self$sampling(c(z_mean, z_log_var))
    list(z_mean, z_log_var, z)
  }
}

Decoder(keras$layers$Layer) %py_class% {
  initialize <- function(original_dim, intermediate_dim = 95, inter_dim2 = 65, name = "decoder", ...) {
    super$initialize(name = name, ...)
    self$dense_proj2 <- layer_dense(units = inter_dim2, activation = "relu")
    self$dense_proj <- layer_dense(units = intermediate_dim, activation = "relu")
    self$dense_output <- layer_dense(units = original_dim, activation = "sigmoid")
  }
  
  call <- function(inputs) {
    x <- self$dense_proj2(inputs)  # Start with the smaller dimension layer
    x <- self$dense_proj(x)  # Then move to the larger dimension layer
    self$dense_output(x)
  }
}

VariationalAutoEncoder(keras$Model) %py_class% {
  initialize <- function(original_dim, intermediate_dim = 95, inter_dim2 = 65, latent_dim = 18, name = "autoencoder", ...) {
    super$initialize(name = name, ...)
    self$original_dim <- original_dim
    self$encoder <- Encoder(latent_dim = latent_dim, intermediate_dim = intermediate_dim, inter_dim2 = inter_dim2)
    self$decoder <- Decoder(original_dim, intermediate_dim = intermediate_dim, inter_dim2 = inter_dim2)
  }
  
  call <- function(inputs) {
    c(z_mean, z_log_var, z) %<-% self$encoder(inputs)
    reconstructed <- self$decoder(z)
    kl_loss <- -0.5 * tf$reduce_mean(z_log_var - tf$square(z_mean) - tf$exp(z_log_var) + 1)
    self$add_loss(kl_loss)
    reconstructed
  }
}

original_dim <- nrow(matrix)
vae <- VariationalAutoEncoder(original_dim, 95, 65, 18)

optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = 1e-3)
vae %>% compile(optimizer, loss = loss_mean_squared_error())
vae %>% fit(
  t(matrix), 
  t(matrix), 
  epochs = 100, # Increase epochs since training might stop early
  batch_size = 64,
  callbacks = list(callback_early_stopping(monitor = 'loss', patience = 5))
)

# Function to extract features using the trained VAE encoder
extract_features <- function(matrix) {
  encoder <- vae$encoder
  features <- encoder(matrix)
  return(features[[3]])
}

RunBEAR <- function(matrix = matrix, coordinates = coordinates, labels = labels, nsize = 3, output_file = "metrics.txt") {
  set.seed(22)
  
  ind <- sample(2, ncol(matrix), replace = TRUE, prob = c(0.7, 0.3))
  
  mat <- matrix[,ind == 1]
  labs <- labels[ind == 1]
  rn <- levels(factor(labs[ind == 1]))
  coords <- coordinates[ind == 1,]
  
  avg_mat <- mat
  
  for (i in 1:length(rn)){
    wc <- coords
    mat_1 <- mat[,which(labs == rn[i])]
    wc <- wc[which(labs == rn[i]),]
    
    for (j in 1:ncol(mat_1)){
      roi <- wc[j,2]
      coi <- wc[j,3]
      allrows <- wc[,2]
      allcols <- wc[,3]
      neighs <- which((allrows %in% c((roi-nsize):(roi+nsize))) & 
                        (allcols %in% c((coi-nsize):(coi+nsize))))
      
      if (length(neighs) < 2){
        next
      }
      
      newj <- rowMeans(mat_1[,neighs])
      avg_mat[,colnames(mat_1)[j]] <- newj
    }
  }
  
  message("Finished training neighborhood averaging")
  
  df <- data.frame(cbind(labs, t(avg_mat)))
  
  for (i in 2:ncol(df)){
    df[,i] <- as.numeric(df[,i])
  }
  
  df$labs <- factor(df$labs)
  train <- df
  
  train_features <- extract_features(as.matrix(train[, -1]))
  train_labels <- as.numeric(train$labs) - 1
  
  classifier <- keras_model_sequential() %>%
    layer_dense(units = 24, activation = 'relu', input_shape = ncol(train_features)) %>%
    layer_dense(units = length(unique(train_labels)), activation = 'softmax')
  
  classifier %>% compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = optimizer_adam(),
    metrics = 'accuracy'
  )
  
  classifier %>% fit(
    train_features, 
    train_labels, 
    epochs = 100, # Increase epochs to allow room for early stopping
    batch_size = 32, 
    validation_split = 0.2,
    callbacks = list(callback_early_stopping(monitor = 'val_loss', patience = 5))
  )
  
  message("Finished model building")
  
  mat <- matrix[,ind == 2]
  labs <- labels[ind == 2]
  coords <- coordinates[ind == 2,]
  
  wc <- coords
  avg_mat <- mat
  mat_1 <- mat
  
  for (j in 1:ncol(mat_1)){
    roi <- wc[j,2]
    coi <- wc[j,3]
    allrows <- wc[,2]
    allcols <- wc[,3]
    neighs <- which((allrows %in% c((roi-nsize):(roi+nsize))) & 
                      (allcols %in% c((coi-nsize):(coi+nsize))))
    
    if (length(neighs) < 2){
      next
    }
    
    newj <- rowMeans(mat_1[,neighs])
    avg_mat[,colnames(mat_1)[j]] <- newj
  }
  
  message("Finished testing neighborhood averaging")
  
  df <- data.frame(cbind(labs, t(avg_mat)))
  
  for (i in 2:ncol(df)){
    df[,i] <- as.numeric(df[,i])
  }
  
  df$labs <- factor(df$labs)
  test <- df
  
  test_features <- extract_features(as.matrix(test[, -1]))
  test_labels <- as.numeric(test$labs) - 1
  
  predictions <- classifier %>% predict(test_features) %>% k_argmax() %>% as.array()
  
  test_labels <- factor(test_labels, levels = 0:(length(unique(labels)) - 1))
  predictions <- factor(predictions, levels = 0:(length(unique(labels)) - 1))
  
  cm <- confusionMatrix(predictions, test_labels)
  
  # Save the confusion matrix and statistics to a file
  sink(output_file)
  print(cm)
  sink() # Stop sinking
  
  return(cm)
}

# Run the function and save the metrics to a file
RunBEAR(matrix = matrix, coordinates = coordinates, labels = labels, nsize = 3, output_file = "~/spatial_cluster_evaluation/metrics.txt")
