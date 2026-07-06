# Linear algebra implementation of a neural network ----

.epochs  <- 1000000
.n_train <- 30
.eta     <- 0.001 # Learning rate
.verbose <- TRUE

# Helpers ----
## Define our activation function ----

# Using a sigmoid/logistic function here. Its derivative uses the result of the
# activation function itself which is computationally convenient.
activation_func <- \(x) 1 / (1 + exp(-x))
activation_func_deriv <- \(x) activation_func(x) * (1 - activation_func(x))

# Data ----
# Need to define our input data and target
set.seed(24601)

#  2 input cols, our XOR inputs - with an extra random noise column
input_mtx <- cbind(
  sample(c(0, 1), .n_train, replace = TRUE),
  sample(c(0, 1), .n_train, replace = TRUE),
  1, # Bias column, which we can't learn XOR without
  runif(.n_train, -1, 1)
)

#  Output is XOR of the first two cols
target_mtx <- xor(input_mtx[, 1], input_mtx[, 2]) |> as.integer()

head(input_mtx)
#>      [,1] [,2] [,3]       [,4]
#> [1,]    1    0    1  0.1184642
#> [2,]    1    1    1 -0.1949854
#> [3,]    0    0    1  0.3469756
#> [4,]    0    1    1  0.3771357
#> [5,]    0    0    1 -0.4886959
#> [6,]    0    0    1  0.6804348

cbind(input_mtx, target_mtx) |> 
  data.frame() |> 
  setNames(c('V1', 'V2', 'Bias', 'Noise', 'Target')) |>
  head()
#>   V1 V2 Bias      Noise Target
#> 1  1  0    1  0.1184642      1
#> 2  1  1    1 -0.1949854      0
#> 3  0  0    1  0.3469756      0
#> 4  0  1    1  0.3771357      1
#> 5  0  0    1 -0.4886959      0
#> 6  0  0    1  0.6804348      0

# Initialise weights ----
# Say we want a 3 x 2 neural net: 2 hidden layers, 3 nodes in each.
# This gives us three layers of weights (input -> h1, h1 -> h2, h2 -> output).

nn_dim <- list(layers = 1, nodes_per_layer = 3)

init_weights <- function(rows, cols) {
  matrix(runif(rows * cols, -0.5, 0.5), nrow = rows, ncol = cols)
}

# set.seed(24601)

# 4 layers in input, so 4 rows. 3 nodes in H1, so 3 cols.
w_ih1 <- init_weights(rows = ncol(input_mtx), cols = nn_dim$nodes_per_layer)

# 3 + 1 (bias) nodes in H1, so 4 rows. 1 output, so 1 col.
w_h1o <- init_weights(rows = nn_dim$nodes_per_layer + 1, cols = 1)


# Perform training loop ----

results <- list()

# For each epoch...
for ( epoch in seq_len(.epochs) ) {
  
  if ( .verbose & (epoch %% 1000) == 0 ) cat(paste('Epoch:', epoch))

  epoch_results <- list(loss = 0)

  # ... and each training example ....
  for ( obs in seq_len(nrow(input_mtx)) ) {

    ## Forward propagation ----
    # First, perform forward propagation, run the input through the weights and:
    # - Cache the activation function results for use in backpropagation
    # - Record the overall outputs so we know how wrong we are, also for backpropagation

    x_i <- matrix(input_mtx[obs, ], nrow = 1)
    y_i <- target_mtx[obs]

    # The value at each node is the activated weighted sum of the inputs,
    #  z_1 = input_1 * w_1 + input_2 * w_2 + input_3 * w_3
    # Which lends itself nicely to matrix multiplication:
    z1   <- activation_func(x_i %*% w_ih1)
    z1b  <- cbind(1, z1) # We need a bias node for layer 1 too
    yhat <- activation_func(z1b %*% w_h1o)

    # Calculate loss and add it to the overall loss for the epoch
    epoch_results[['loss']] <- epoch_results[['loss']] + (y_i - yhat)^2

    ## Backpropagation ----

    # This gives us the direction in which to update our weights - i.e. which
    # direction should we step in to yield the largest decrease in loss.

    # Output to last hidden layer backprop
    # error * sensitivity * activity = (y - y_hat) * [y_hat * (1 - y_hat)] * z_j
    #                                = (y - y_hat) * activation_func_deriv(y_hat) * z_j
    delta_o <- (y_i - yhat) * activation_func_deriv(yhat)

    # First hidden layer to input layer backprop
    delta_h1_full <- activation_func_deriv(z1b) * (delta_o %*% t(w_h1o))
    delta_h1      <- delta_h1_full[, -1, drop = FALSE]   # drop bias column


    ## Weight updates ----

    # w = w + η * δ * activity_of_sending_node

    w_h1o <- w_h1o + .eta * (t(z1b) %*% delta_o)
    w_ih1 <- w_ih1 + .eta * (t(x_i) %*% delta_h1)

  }

  # Assign this epoch's loss back to results list
  # Capture mean absolute weight per input column, alongside loss
  results[[epoch]] <- data.frame(
    epoch        = epoch,
    loss         = 0.5 * epoch_results[['loss']],
    weight_x1    = mean(abs(w_ih1[1, ])),  # first XOR col
    weight_x2    = mean(abs(w_ih1[2, ])),  # second XOR col
    weight_bias  = mean(abs(w_ih1[3, ])),  # bias col
    weight_noise = mean(abs(w_ih1[4, ]))   # noise col
  )

  if ( .verbose & (epoch %% 1000) == 0 ) cat(paste(' | Loss:', 0.5 * epoch_results[['loss']]), '\n')

}

loss_df <- do.call(rbind, results)

# Graphing results ----

library(ggplot2)
library(dplyr)
library(tidyr)

plot_dim <- list(width = 1684, height = 1684 * 9/16)
dir.create('plots', showWarnings = FALSE)

sysfonts::font_add_google('Space Grotesk', "space_grotesk") 
showtext::showtext_auto()

palette <- list(
  'teal' = '#14b8a6',
  'pink' = '#ec4899',
  'grey' = '#888888'
)

loss_df %>% 
  ggplot(aes(x = epoch, y = loss)) +
  geom_line() +
  scale_x_log10(
    breaks = c(1, 10, 100, 1000, 10000, 100000, 1000000),
    labels = scales::comma,
    limits = c(1, 1200000)
  ) +
  labs(
    title = 'Total loss by epoch',
    x = 'log(Epoch)', y = 'Total L2 Loss'
  ) +
  theme_minimal(base_family = "space_grotesk")

ggsave(  
  file.path('plots', "nn-loss.jpg"),  
  device = 'jpg',  
  width = plot_dim$width,  
  height = plot_dim$height,  
  units = 'px',  
  dpi = 250
)

# loss_df %>% 
#   select(epoch, starts_with('weight_')) %>% 
#   pivot_longer(-epoch, names_to = 'input', values_to = 'weight_value') %>% 
#   ggplot(aes(x = epoch, y = weight_value, colour = input)) +
#   geom_line() +
#   scale_x_log10(labels = scales::comma)

# XOR demonstration
cbind(input_mtx, target_mtx) %>% 
  as_tibble() %>% 
  mutate(target_mtx = as.character(target_mtx)) %>% 
  ggplot(aes(x = V1, y = V2, colour = target_mtx)) +
  geom_jitter(alpha = 0.7, width = 0.025, height = 0.025) +
  labs(
    title = 'A demonstration of XOR data',
    subtitle = 'Jitter applied for demonstration',
    caption = 'n = 30',
    x = 'Col 1', y = 'Col 2', colour = 'Class'
  ) +
  coord_cartesian(xlim = c(-0.25, 1.25), ylim = c(-0.25, 1.25)) +
  theme_minimal(base_family = "space_grotesk") +
  theme(legend.position = 'bottom')

ggsave(  
  file.path('plots', "xor-data.jpg"),  
  device = 'jpg',  
  width = plot_dim$width,  
  height = plot_dim$width,  
  units = 'px',  
  dpi = 250
)
