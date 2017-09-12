library(tensorflow)
library(reticulate)
library(argparse)

contrib <- tf$contrib
layers <- contrib$layers

# Training settings
parser <- ArgumentParser(description="MNIST example")


parser$add_argument("--batch-size", type="integer", default=64L, 
                    help="input number of batch size for training (default: 64L)",
                    metavar="number")

parser$add_argument("--test-batch-size", type="integer", default=1000L, 
                    help="input number of batch size for testing (default: 1000L)",
                    metavar="number")

parser$add_argument("--epochs", type="integer", default=10L, 
                    help="number of epochs to train (default: 10L)",
                    metavar="number")


