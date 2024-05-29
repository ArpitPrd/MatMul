import torch
import torch.nn as nn
import torch.optim as optim

# Generate data
data_size = 100
num_el = 2 * 2 * 2 * data_size
input_data = torch.randint(1, 4, (data_size, 2, 2, 2)).float()  # Simplified data generation
input_data = input_data.cuda()
# Prepare output data (matrix multiplication)
output_data = torch.stack([x[0] @ x[1] for x in input_data])
output_data = output_data.cuda()
# Define a more complex model
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
model.cuda()
# Define loss and optimizer
mse_loss = nn.MSELoss()
mse_loss.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 10000000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output_pred = model(input_data)
    loss = mse_loss(output_pred, output_data)
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Evaluate final loss
model.eval()
with torch.no_grad():
    final_output = model(input_data)
    final_loss = mse_loss(final_output, output_data)
    print(f"Final Loss: {final_loss.item()}")
t1 = torch.tensor([[1., 0.], [0., 1.]])
t2 = torch.tensor([[2., 1.], [1., 2.]])
t = torch.stack([t1, t2]).cuda()
out = model(torch.unsqueeze(t, 0))
print(out)