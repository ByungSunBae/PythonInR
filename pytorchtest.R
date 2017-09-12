## Using reticulate and pytorch in Rstudio
# MNIST classification
# Reference : 
#  1) https://github.com/pytorch/examples/blob/master/mnist/main.py
#  2) https://github.com/longhowlam/ReticulateTestDrive/blob/master/pytorch.R

library(argparse)
library(reticulate)

torch <- import("torch")
Sys.sleep(2)
nn <- torch$nn
Sys.sleep(2)
Variable <- torch$autograd$Variable
Sys.sleep(2)
Func <- torch$nn$functional
Sys.sleep(2)
Optim <- torch$optim

torchvision <- import("torchvision")
Sys.sleep(2)
DTsets <- torchvision$datasets
Sys.sleep(2)
Transforms <- torchvision$transforms

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

parser$add_argument("--lr", default=0.01, type="double",
                    metavar="LR",
                    help="learning rate (default: 0.01)")

parser$add_argument("--momentum", default=0.5, type="double",
                    metavar="M",
                    help="SGD momentum (default: 0.01)")

parser$add_argument("--no-cuda", action="store_true", default=TRUE,
                    help="disables CUDA training")

parser$add_argument("--seed", type="integer", default=1L, 
                    help="random seed (default: 1L)",
                    metavar="S")

parser$add_argument("--log-interval", type="integer", default=10L, 
                    help="How many batches to wait before logging training status",
                    metavar="N")

args <- parser$parse_args()
args$cuda <- !(args$no_cuda) & torch$cuda$is_available()

torch$set_num_threads(8L)

torch$manual_seed(as.integer(args$seed))
if (args$cuda)
  torch$cuda$manual_seed(as.integer(args$seed))

Sys.sleep(2)
train_loader <- torch$utils$data$DataLoader(
  DTsets$MNIST("../data", train=TRUE, download=TRUE,
               transform=Transforms$Compose(
                 c(Transforms$ToTensor(),
                   Transforms$Normalize(list(0.1307), list(0.3081)))
               )),
  batch_size=as.integer(args$batch_size), shuffle=TRUE, num_workers=1L, pin_memory=TRUE)

Sys.sleep(2)
test_loader <- torch$utils$data$DataLoader(
  DTsets$MNIST("../data", train=FALSE, 
               transform=Transforms$Compose(
                 c(Transforms$ToTensor(),
                   Transforms$Normalize(list(0.1307), list(0.3081)))
               )),
  batch_size=as.integer(args$test_batch_size), shuffle=TRUE, num_workers=1L, pin_memory=TRUE)


Net <- py_run_string(
"
import torch.nn as nn
import torch.nn.functional as F
  
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
  
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
")

model = Net$Net()
if (args$cuda)
  model$cuda()

optimizer = Optim$SGD(model$parameters(), lr=args$lr, momentum=args$momentum)

py = import_builtins()

zz = py$enumerate(train_loader)
qq = iterate(zz, simplify = FALSE)
Sys.sleep(2)
### Train ###
model$train()
for (epoch in 1:3){
  i=0
  for (a in qq)
  {
    i=i+1
    images = Variable(a[[2]][[1]])
    labels = Variable(a[[2]][[2]])
    
    # Forward + Backward + Optimize
    optimizer$zero_grad()
    outputs = model(images)
    loss = Func$nll_loss(outputs, labels)
    loss$backward()
    optimizer$step()
    
    if (i%%100 == 0){
      cat("epoch ");cat(epoch); cat(" / ");cat(as.integer(args$epochs));
      cat(" step "); cat(i+1)
      cat(" loss: ");cat(loss$data$numpy())
      cat("\n")
      
    }
  }
}

Sys.sleep(2)
### Test ###
model$eval()
correct = 0
total   = 0

zz = py$enumerate(test_loader)
qq = iterate(zz, simplify = FALSE)

for (a in qq){
  images = Variable(a[[2]][[1]])
  labels = Variable(a[[2]][[2]])
  outputs = model(images)
  predicted = torch$max(outputs$data, 1L)
  correct = correct + 
    sum(  as.vector(predicted[[2]]$numpy()) == labels$data$numpy())
}

sprintf("Accuracy of the model on the 10000 test images: %f  ", correct/10000)

























