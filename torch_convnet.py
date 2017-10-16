import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# It enables benchmark mode in cudnn.
# benchmark mode is good whenever your input sizes for your network do not vary.
# This way, cudnn will look for the optimal set of algorithms for that
# particular configuration (which takes some time).
# This usually leads to faster runtime.
torch.backends.cudnn.benchmark = True

batch_size = 50
num_tries = 100
random_data = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=7, stride=1, bias=False, padding=3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=7, stride=1, bias=False, padding=3)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=7, stride=1, bias=False, padding=3)
        self.conv4 = nn.Conv2d(3, 16, kernel_size=7, stride=2, bias=False, padding=3)
        self.conv5 = nn.Conv2d(16, 32, kernel_size=7, stride=2, bias=False, padding=3)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=7, stride=1, bias=False, padding=3)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False, padding=1)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False, padding=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, stride=2, bias=False, padding=1)
        self.conv10 = nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False, padding=1)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=2, bias=False, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x

model = Net()
model.cuda()
model.eval()

# Warmup
if random_data:
    _inp = np.random.randn(batch_size, 3, 300, 300)
    data = Variable(torch.FloatTensor(_inp))
else:
    data = Variable(torch.ones(batch_size, 3, 300, 300))

data = data.cuda()
model(data)

for i in range(num_tries):
    t0 = time.time()
    for j in range(10):
        if random_data:
            _inp = np.random.randn(batch_size, 3, 300, 300)
            data = Variable(torch.FloatTensor(_inp))
            data = data.cuda()
        model(data)
    t1 = time.time()
    elapsed_time = t1- t0
    print(elapsed_time)
    if i == 0:
        total_time = elapsed_time
    else:
        total_time = 0.9 * total_time + 0.1 * elapsed_time

print("")
print("===============================")
print("===============================")
print("")
print("Avg time (miliseconds):", total_time * 1000)
