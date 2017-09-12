########################################################
# Rstudio에서 reticulate를 활용하여 간단한 gan 구현하기#
########################################################
library(data.table)
library(reticulate)
library(tensorflow)
library(ggplot2)
library(MASS)

####
# Data : https://www.dropbox.com/s/ryy436ymuyzqbr0/creditcard.csv?dl=0
#### download and unzip!

# if (!(dir.exists("./data")))
#   dir.create("./data")
# 
# download.file("https://www.dropbox.com/s/ryy436ymuyzqbr0/creditcard.csv?dl=0",
#               "./data/creditcard.csv")
# and "$ mv creditcard.csv?dl=0 creditcard.csv" in terminal

FraudCredit <- fread("data/creditcard.csv")
FraudCredit[, c("Time", "Amount", "Class") := list(NULL, NULL, NULL)]

FraudCredit <- FraudCredit[sample(1:nrow(FraudCredit), size = 10000)]

MeanVecs <- apply(FraudCredit, 2, mean)
SDVecs <- apply(FraudCredit, 2, sd)
CovMat <- cov(FraudCredit)

FraudCredit <- as.data.table(scale(FraudCredit))

summary(FraudCredit)
# FraudCredit


tf <- import("tensorflow")
contrib <- tf$contrib
np <- import("numpy")

config <- tf$ConfigProto
config$allow_soft_placement = TRUE
# TargetDistDT <- data.table(X = xs, PDFvalue = dnorm(x = xs, mean = mu, sd = sigma))


tf$reset_default_graph()
# 분포확인
# plot(TargetDistDT[, X], 
#      TargetDistDT[, PDFvalue], 
#      type="l", xlab = "X", ylab = "density of noraml")
ggplot(FraudCredit, aes(x = (V1 - mean(V1))/sd(V1))) + geom_density()

TrainIters <- 10000
M <- 200 # minibatch size

xavier_init <- function(size){
  in_dim = size[[1]]
  xavier_stddev = 1. / tf$sqrt(in_dim / 2.)
  return(tf$random_normal(shape=size, stddev=xavier_stddev))
}


X = tf$placeholder(tf$float32, shape(NULL, 28))

D_W1 = tf$Variable(xavier_init(shape(28, 32)))
D_b1 = tf$Variable(tf$zeros(shape(32)))

D_W2 = tf$Variable(xavier_init(shape(32, 1)))
D_b2 = tf$Variable(tf$zeros(shape(1)))

theta_D = c(D_W1, D_W2, D_b1, D_b2)


Z = tf$placeholder(tf$float32, shape(NULL, 28))

G_W1 = tf$Variable(xavier_init(shape(28, 42)))
G_b1 = tf$Variable(tf$zeros(shape(42)))

G_W2 = tf$Variable(xavier_init(shape(42, 28)))
G_b2 = tf$Variable(tf$zeros(shape(28)))

theta_G = c(G_W1, G_W2, G_b1, G_b2)


# generator <- function(z){
#   G_h1 = tf$nn$relu(tf$matmul(z, G_W1) + G_b1)
#   G_log_prob = tf$matmul(G_h1, G_W2) + G_b2
#   G_prob = tf$nn$sigmoid(G_log_prob)
#   return(G_prob)
# }

# discriminator <- function(x){
#   D_h1 = tf$nn$relu(tf$matmul(x, D_W1) + D_b1)
#   D_logit = tf$matmul(D_h1, D_W2) + D_b2
#   D_prob = tf$nn$sigmoid(D_logit)
#   return(c(D_prob, D_logit))
# }

### least square gan
generator <- function(z){
  G_h1 = tf$nn$relu(tf$matmul(z, G_W1) + G_b1)
  G_logit = tf$matmul(G_h1, G_W2) + G_b2
  G_out = tf$nn$tanh(G_logit)
  return(G_out)
}

discriminator <- function(x){
  D_h1 = tf$nn$relu(tf$matmul(x, D_W1) + D_b1)
  D_logit = tf$matmul(D_h1, D_W2) + D_b2
  # D_prob = tf$nn$sigmoid(D_logit)
  return(D_logit)
}

G_sample = generator(Z)
# D_tmp = discriminator(X)
D_real = discriminator(X)
# D_logit_real = D_tmp[[2]]

# D_tmp2 = discriminator(G_sample)
D_fake = discriminator(G_sample)
# D_logit_fake = D_tmp2[[2]]

# add 1e-10 to logs, otherwise we'll get nan during training
# D_loss = -tf$reduce_mean(tf$log(1e-10 + D_real) + tf$log(1. - D_fake))
# G_loss = -tf$reduce_mean(tf$log(1e-10 + D_fake))
D_loss = 0.5 * (tf$reduce_mean(tf$pow((D_real - 1), 2)) + tf$reduce_mean(tf$pow(D_fake, 2)))
G_loss = 0.5 * tf$reduce_mean(tf$pow((D_fake - 1), 2))

# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf$train$AdamOptimizer()$minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf$train$AdamOptimizer()$minimize(G_loss, var_list=theta_G)

sample_z <- function(m){
  return(mvrnorm(n = m, mu = MeanVecs, Sigma = CovMat))
}

sample_z(mb_size)

mb_size = 100

FraudCredit_mat <- as.matrix(FraudCredit)

with(tf$Session() %as% sess,{
  sess$run(tf$global_variables_initializer())
  
  k1 = 1
  k2 = 1
  
  for (i in 1:500){
    for (it1 in 1:k1){
      for (j in 1:mb_size){
        idx <- ((mb_size*(j-1)):(mb_size*j))[-1]
        X_mb <- FraudCredit_mat[idx,]
        DD_tmp = sess$run(c(D_solver, D_loss), 
                          feed_dict = dict(X = X_mb, Z = sample_z(mb_size)))
        D_loss_curr = DD_tmp[[2]]
        
        
      }
    }
    for (it2 in 1:k2){
      for (j in 1:mb_size){
        idx <- ((mb_size*(j-1)):(mb_size*j))[-1]
        X_mb <- FraudCredit_mat[idx,]
        GG_tmp = sess$run(c(G_solver, G_loss),
                          feed_dict = dict(Z = sample_z(mb_size)))
        
        G_loss_curr = GG_tmp[[2]]
      }
    }
    
    
    if (i %% 100 == 0){
      print(paste(cat("Iter:", i, ", D loss : ", D_loss_curr, ", G loss : ", G_loss_curr, "\n")))
    }
  }
  Gen = generator(Z)
  sess$run(tf$global_variables_initializer())
  GenData <- sess$run(Gen, feed_dict = dict(Z = sample_z(mb_size)))
  assign("GenData", as.data.table(GenData), envir = .GlobalEnv)
})

ggplot() + 
  geom_density(data = FraudCredit, aes(x = V7)) + 
  geom_density(data = GenData, aes(x = V7), color = "blue")

