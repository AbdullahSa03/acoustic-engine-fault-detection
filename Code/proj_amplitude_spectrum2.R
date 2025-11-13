wav_to_amplitude_spectra_classifier = function(main_folder, output_folder = NULL, 
                                               window_size = 1024, normalize = TRUE,
                                               use_parallel = TRUE, num_cores = NULL,
                                               save_plots = FALSE) { # Added option to skip plotting
  # Load required libraries
  if (!require("tuneR")) install.packages("tuneR")
  if (!require("tools")) install.packages("tools")
  library(tuneR)
  library(tools)
  
  # Create output folder if specified
  if (!is.null(output_folder) && !dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
  
  # Get all WAV files with their classes
  all_wav_files = c()
  class_folders = list.dirs(main_folder, full.names = TRUE, recursive = FALSE)
  
  for (class_folder in class_folders) {
    class_name = basename(class_folder)
    wav_files = list.files(class_folder, pattern = "\\.wav$", full.names = TRUE)
    
    if (length(wav_files) > 0) {
      class_data = data.frame(
        file_path = wav_files,
        class = rep(class_name, length(wav_files)),
        stringsAsFactors = FALSE
      )
      all_wav_files = rbind(all_wav_files, class_data)
    }
  }
  
  cat("Found", nrow(all_wav_files), "WAV files across", length(class_folders), "classes\n")
  
  # Define the processing function
  process_wav = function(i) {
    file_path = all_wav_files$file_path[i]
    class_name = all_wav_files$class[i]
    file_name = basename(file_path)
    base_name = file_path_sans_ext(file_name)
    
    tryCatch({
      # Read wave file - more efficiently
      wav_data = readWave(file_path)
      
      # Get audio data
      audio_data = wav_data@left
      fs = wav_data@samp.rate
      
      # Compute FFT more efficiently
      n = length(audio_data)
      # Use window_size if specified, otherwise use next power of 2
      if (window_size > 0) {
        nfft = window_size
      } else {
        nfft = 2^ceiling(log2(n))
      }
      
      # Apply FFT
      fft_result = fft(audio_data)/n
      half = floor(n/2) + 1
      fft_half = fft_result[1:half]
      
      # Calculate amplitude spectrum
      amplitude_spectrum = Mod(fft_half)
      
      # Normalize if requested
      if (normalize && max(amplitude_spectrum) > 0) {
        amplitude_spectrum = amplitude_spectrum / max(amplitude_spectrum)
      }
      
      # Create frequency axis
      freq_axis = seq(0, fs/2, length.out = half)
      
      # Key and result
      spectrum_key = paste0(class_name, "_", base_name)
      
      # Return only essential data (minimize data transfer between workers)
      return(list(
        key = spectrum_key,
        amplitude = amplitude_spectrum,
        freq = freq_axis,
        fs = fs,
        class = class_name,
        file = file_name
      ))
    }, error = function(e) {
      # Return NULL on error
      return(NULL)
    })
  }
  
  # Process files
  results = NULL
  
  if (use_parallel && nrow(all_wav_files) > 1) {
    if (is.null(num_cores)) {
      if (!require("parallel")) install.packages("parallel")
      library(parallel)
      num_cores = detectCores() - 1
      if (num_cores < 1) num_cores = 1
    }
    
    cat("Processing", nrow(all_wav_files), "files using", num_cores, "cores...\n")
    
    # Use foreach for better progress tracking
    if (!require("foreach")) install.packages("foreach")
    if (!require("doParallel")) install.packages("doParallel")
    library(foreach)
    library(doParallel)
    
    # Register parallel backend
    cl = makeCluster(num_cores)
    registerDoParallel(cl)
    
    # Process files in parallel with progress reporting
    results = foreach(i = 1:nrow(all_wav_files), 
                      .packages = c("tuneR", "tools"),
                      .combine = 'c') %dopar% {
                        result = process_wav(i)
                        if (i %% 10 == 0) cat(".")  # Simple progress indicator
                        if (i %% 100 == 0) cat(i, "files processed\n")
                        list(result)  # Wrap in list for proper combining
                      }
    
    stopCluster(cl)
    cat("\nParallel processing complete\n")
  } else {
    # Sequential processing
    cat("Processing files sequentially...\n")
    results = vector("list", nrow(all_wav_files))
    for (i in 1:nrow(all_wav_files)) {
      results[[i]] = process_wav(i)
      if (i %% 10 == 0) cat(".")
      if (i %% 100 == 0) cat(i, "files processed\n")
    }
    cat("\nSequential processing complete\n")
  }
  
  # Remove NULL results
  results = results[!sapply(results, is.null)]
  
  # Organize results
  all_spectra = list()
  all_metadata = data.frame(
    file_path = character(),
    class = character(),
    file_name = character(),
    stringsAsFactors = FALSE
  )
  
  # Batch process results
  for (result in results) {
    if (!is.null(result)) {
      all_spectra[[result$key]] = list(
        amplitude = result$amplitude,
        frequency = result$freq,
        sampling_rate = result$fs,
        class = result$class
      )
      
      all_metadata = rbind(all_metadata, data.frame(
        file_path = all_wav_files$file_path[which(basename(all_wav_files$file_path) == result$file)],
        class = result$class,
        file_name = result$file,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Save outputs only if requested (do this outside the parallel loop)
  if (!is.null(output_folder)) {
    # Save all spectra data efficiently
    cat("Saving results to disk...\n")
    
    # Save metadata
    write.csv(all_metadata, file.path(output_folder, "metadata.csv"), row.names = FALSE)
    
    # Create class folders
    unique_classes = unique(all_metadata$class)
    for (class_name in unique_classes) {
      class_dir = file.path(output_folder, class_name)
      if (!dir.exists(class_dir)) dir.create(class_dir)
    }
    
    # Save individual spectra data and plots
    if (save_plots) {
      for (spec_name in names(all_spectra)) {
        spec = all_spectra[[spec_name]]
        class_name = spec$class
        base_name = gsub(paste0("^", class_name, "_"), "", spec_name)
        
        # Save CSV
        write.csv(
          data.frame(
            frequency = spec$frequency,
            amplitude = spec$amplitude
          ),
          file = file.path(output_folder, class_name, paste0(base_name, "_spectrum.csv")),
          row.names = FALSE
        )
        
        # Generate plot
        png(file.path(output_folder, class_name, paste0(base_name, "_spectrum.png")), 
            width = 800, height = 600)
        plot(spec$frequency, spec$amplitude, type = "l",
             xlab = "Frequency (Hz)", ylab = "Amplitude",
             main = paste("Amplitude Spectrum -", class_name, "-", base_name))
        dev.off()
      }
    }
  }
  
  cat("Processing complete. Processed", length(results), "files successfully.\n")
  return(list(
    spectra = all_spectra,
    metadata = all_metadata
  ))
}

# Function to prepare the dataset for machine learning
prepare_training_data = function(spectra_data) {
  # Extract spectra and classes
  spectra_list = spectra_data$spectra
  metadata = spectra_data$metadata
  
  cat("Preparing training data from", length(spectra_list), "spectra...\n")
  
  # Determine the required length for all spectra (use the maximum)
  max_length = max(sapply(spectra_list, function(x) length(x$amplitude)))
  
  cat("Standardizing all spectra to length:", max_length, "\n")
  
  # Create a matrix to hold all spectra
  data_matrix = matrix(0, nrow = length(spectra_list), ncol = max_length)
  
  # Extract class labels in the same order as the spectra
  labels = character(length(spectra_list))
  
  # Fill the matrix with amplitude values
  i = 1
  for (spec_name in names(spectra_list)) {
    spec = spectra_list[[spec_name]]
    amp_data = spec$amplitude
    
    # Resample if needed to ensure consistent length
    if (length(amp_data) != max_length) {
      amp_data = approx(1:length(amp_data), amp_data, n = max_length)$y
    }
    
    data_matrix[i, ] = amp_data
    labels[i] = spec$class
    i = i + 1
  }
  
  # Convert to dataframe and add labels
  training_df = as.data.frame(data_matrix)
  names(training_df) = paste0("freq_bin_", 1:ncol(training_df))
  training_df$class = factor(labels)
  
  cat("Training data prepared with", nrow(training_df), "samples and", 
      ncol(training_df)-1, "features\n")
  
  return(training_df)
}


result = wav_to_amplitude_spectra_classifier(
  main_folder = "IDMT-ISA-ELECTRIC-ENGINE/train_cut", 
  output_folder = "IDMT-ISA-ELECTRIC-ENGINE/results",
  use_parallel = TRUE
)

# Prepare data for training
train_data = prepare_training_data(result)

test_result = wav_to_amplitude_spectra_classifier(
  main_folder = "IDMT-ISA-ELECTRIC-ENGINE/test_cut", 
  output_folder = "IDMT-ISA-ELECTRIC-ENGINE/results",
  use_parallel = TRUE
)

test_data = prepare_training_data(test_result)


# Determine how many components to keep
pca_result <- prcomp(train_data[, -ncol(train_data)], scale. = TRUE)
var_explained <- cumsum(pca_result$sdev^2 / sum(pca_result$sdev^2))
num_components <- which(var_explained >= 0.95)[1]  # Keep 95% of variance
num_components
# Project data onto PCA space
train_pca <- predict(pca_result, train_data[, -ncol(train_data)])[, 1:num_components]
train_pca_df <- as.data.frame(train_pca)
train_pca_df$label <- train_data$class

test_pca <- predict(pca_result, test_data[, -ncol(test_data)])[, 1:num_components]
test_pca_df <- as.data.frame(test_pca)
test_pca_df$label <- test_data$class


library(randomForest)
library(e1071)
library(rpart)
library(xgboost)

bag.model=randomForest(label~.,data=train_pca_df,mtry=146,importance=TRUE)
yhat.bag = predict(bag.model,newdata=test_pca_df, type="class")
print("Bagging Confusion Matrix:")
table(yhat.bag,test_pca_df$label)

bag_error_rate = mean(yhat.bag!=test_pca_df$label)
print(paste("Bagging Error Rate:", round(bag_error_rate, 4)))


#p=146, sqrt(146) approx.=13
rf.model=randomForest(label~.,data=train_pca_df,mtry=70,importance=TRUE)
yhat.rf = predict(rf.model,newdata=test_pca_df, type="class")
print("Random Forest Confusion Matrix:")
table(yhat.rf,test_pca_df$label)
rf_error_rate = mean(yhat.rf!=test_pca_df$label)
print(paste("Random Forest Error Rate:", round(rf_error_rate, 4)))
varImpPlot(rf.model)

# 1. Support Vector Machine (SVM)
svm.model <- svm(label ~ ., data = train_pca_df, kernel = "radial", 
                 cost = 10, gamma = 0.1, probability = TRUE)
yhat.svm <- predict(svm.model, newdata = test_pca_df)
svm_conf_matrix <- table(yhat.svm, test_pca_df$label)
print("SVM Confusion Matrix:")
print(svm_conf_matrix)
svm_error_rate <- mean(yhat.svm != test_pca_df$label)
print(paste("SVM Error Rate:", round(svm_error_rate, 4)))

# 3. Decision Tree
dt.model <- rpart(label ~ ., data = train_pca_df, method = "class")
yhat.dt <- predict(dt.model, newdata = test_pca_df, type = "class")
dt_conf_matrix <- table(yhat.dt, test_pca_df$label)
print("Decision Tree Confusion Matrix:")
print(dt_conf_matrix)
dt_error_rate <- mean(yhat.dt != test_pca_df$label)
print(paste("Decision Tree Error Rate:", round(dt_error_rate, 4)))

# 4. XGBoost
# Convert labels to numeric for XGBoost
label_mapping <- as.integer(factor(train_pca_df$label)) - 1
test_label_mapping <- as.integer(factor(test_pca_df$label)) - 1

# Prepare matrices for XGBoost
train_matrix <- as.matrix(train_pca_df[, -which(names(train_pca_df) == "label")])
test_matrix <- as.matrix(test_pca_df[, -which(names(test_pca_df) == "label")])

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = train_matrix, label = label_mapping)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label_mapping)

# Set parameters
params <- list(
  objective = "multi:softmax",
  num_class = length(unique(train_pca_df$label)),
  eta = 0.3,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train XGBoost model
xgb.model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# Make predictions
yhat.xgb <- predict(xgb.model, dtest)
# Convert numeric predictions back to original labels
class_levels <- levels(factor(train_pca_df$label))
yhat.xgb <- class_levels[yhat.xgb + 1]
xgb_conf_matrix <- table(yhat.xgb, test_pca_df$label)
print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)
xgb_error_rate <- mean(yhat.xgb != test_pca_df$label)
print(paste("XGBoost Error Rate:", round(xgb_error_rate, 4)))


# 5. K-Nearest Neighbors
knn.model <- train(
  label ~ ., 
  data = train_pca_df,
  method = "knn",
  trControl = trainControl(method = "cv", number = 5, allowParallel = FALSE),
  tuneGrid = data.frame(k = c(3, 5, 7, 9, 11))
)
yhat.knn <- predict(knn.model, newdata = test_pca_df)
knn_conf_matrix <- table(yhat.knn, test_pca_df$label)
print("KNN Confusion Matrix:")
print(knn_conf_matrix)
knn_error_rate <- mean(yhat.knn != test_pca_df$label)
print(paste("KNN Error Rate:", round(knn_error_rate, 4)))

# Compare all models
error_rates <- c(
  "Bagging" = bag_error_rate,
  "Random Forest" = rf_error_rate,
  "SVM" = svm_error_rate,
  "Decision Tree" = dt_error_rate,
  "XGBoost" = xgb_error_rate,
  "KNN" = knn_error_rate
)

accuracy_rates <- 1 - error_rates
print("Model Comparison (Accuracy):")
print(round(accuracy_rates * 100, 2))

# Create a simple barplot of accuracies
barplot(sort(accuracy_rates), 
        main = "Model Accuracy Comparison", 
        xlab = "Model", ylab = "Accuracy", 
        col = "steelblue",
        ylim = c(0, 1),
        las = 2)
abline(h = seq(0, 1, 0.1), col = "gray", lty = 2)






