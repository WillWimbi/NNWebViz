#!/usr/bin/env python3
# simple_grad_demo.py
#
# fc1: 2 → 10
# fc2: 10 → 5
# ReLU
# fc3: 5 → 1
#
# One forward pass on x=[1.5, 2.5], squared‑error to y=3.75,
# then backward; prints all activations and grads.

import torch
torch.manual_seed(42)          # reproducible, feel free to remove

class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = torch.nn.Linear(2, 10, bias=True)
        self.fc2  = torch.nn.Linear(10, 5, bias=True)
        self.relu = torch.nn.ReLU()
        self.fc3  = torch.nn.Linear(5, 1, bias=True)

    def forward(self, x):
        z1 = self.fc1(x)          # 10‑vector
        z2 = self.fc2(z1)         # 5‑vector
        a2 = self.relu(z2)        # ReLU
        yhat = self.fc3(a2)       # scalar
        return yhat, (z1, z2, a2)
import torch.optim
def run_once():
    net = TinyNet()
    x = torch.tensor([1.5, 2.5]).float()
    y = torch.tensor([3.75]).float()

    # ---- forward ----
    yhat, (z1, z2, a2) = net(x)
    loss = (yhat - y).pow(2).sum()
    # Print initialized weights and biases
    print("\n=== Initialized Parameters ===")
    for name, param in net.named_parameters():
        print(f"{name} shape {tuple(param.shape)}:")
        print(param.detach().numpy())

    print("\n=== Forward ===")
    print("input x:", x.tolist())
    print("z1 (fc1 out):", z1.detach().tolist())
    print("z2 (fc2 out):", z2.detach().tolist())
    print("a2 (after ReLU):", a2.detach().tolist())
    print("ŷ :", yhat.item())
    print("loss:", loss.item())

    # ---- backward ----
    loss.backward()

    print("\n=== Gradients ===")
    for name, param in net.named_parameters():
        print(f"{name}.grad shape {tuple(param.grad.shape)}")
        print(param.grad.detach().numpy())


if __name__ == "__main__":
    run_once()

print(torch.randn(1,5,4,5)+torch.randn(1,1,4,5))