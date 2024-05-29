print('import start')
import torch
import torch.nn as nn
# import torch.optim as optim
# from matmul import MatrixMultiplicationModel
print('import complete')
# print("in testing")
class MatrixMultiplicationModel(nn.Module):
    def __init__(self):
        super(MatrixMultiplicationModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, bias=True)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 2 * 2, 16)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = x.view(-1, 2, 2)  # Reshape to (batch_size, 2, 2)
        return x

model = MatrixMultiplicationModel()
checkpoint = torch.load('matrix_multiplication_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()  
model.cuda()
print(f"Model loaded. Last epoch: {epoch}, Last loss: {loss}")

t1 = torch.tensor([[1., 0.], [0., 1.]])
t2 = torch.tensor([[1., 0.], [0., 1.]])
t = torch.stack([t1, t2]).cuda()
out = model(torch.unsqueeze(t, 0))
print(t1 @ t2)
print(out)
