import torch
import torch.nn as nn
r=torch.tensor([1,2,3])

innie1=torch.tensor([[[3,6,1,2],[3,6,9,9]],[[3,6,1,2],[3,6,9,9]]],dtype=torch.float32)
innie2=torch.tensor([5,6,7,8],dtype=torch.float32)
outty = torch.matmul(innie1,innie2)
print("matmully!",outty)
# Create a tensor of ones with the same shape as outty and set it as the gradient
outty.requires_grad_(True)  # Make sure outty requires gradients
gradient = torch.ones_like(outty)  # Create gradient tensor
outty.backward(gradient)  # Set all gradients to 1
print("matmully graddy!",outty.grad)
print("outty parents graddies!",)

# Create tensors with shapes (20,3,2) and (5,4,3,2)
tensor1 = torch.randn(1,20, 3, 2)  # Random tensor of shape (20,3,2)
tensor2 = torch.randn(5, 4, 2, 3).reshape(1,20,2,3)  # Random tensor of shape (5,4,3,2)
# These tensors are matmul-able because PyTorch will broadcast the batch dimensions
# 5,4 will be treated as equivalent to 20 (5*4=20) for broadcasting purposes
result = tensor1 @ tensor2  # This performs matrix multiplication

# Print shapes to verify
print(f"Shape of tensor1: {tensor1.shape}")
print(f"Shape of tensor2: {tensor2.shape}")
print(f"Shape of result: {result.shape}")  # Expected shape: (5, 4, 3, 3)
print("this is the result of the matmul 4dim @ 3dim!!",(torch.randn(7,1,5,4) @ torch.randn(3,4,5)).shape)#--> 7,3,5,4 @ 1,3,4,5
print("this is 3dim @ 3dim",(torch.randn(3,4,5) @ torch.randn(3,5,4)).shape)
# print("Dividing tensors:",torch.tensor([1,2,3]).div(torch.tensor([2,2,2])))
# new = torch.tensor([[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]) @ torch.tensor([[[4,5,6,4,5],[4,5,6,4,5],[4,5,6,4,5]],[[4,5,6,4,5],[4,5,6,4,5],[4,5,6,4,5]]])
# print("different sizes of matmuls",new)

# # Create a tensor with the same shape as r but filled with zeros
# zeros_like_r = torch.zeros_like(r)
# print("Original tensor r:", r)
# print("Zeros like r:", zeros_like_r)

# # Example with a different tensor shape
# matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
# zeros_like_matrix = torch.zeros_like(matrix)
# print("\nOriginal matrix:", matrix)
# print("Zeros like matrix:", zeros_like_matrix)

# # You can also specify a different dtype
# zeros_like_float = torch.zeros_like(r, dtype=torch.float32)
# print("\nZeros like r with float32 dtype:", zeros_like_float)

# # Or create ones instead of zeros with similar syntax
# ones_like_r = torch.ones_like(r)
# print("Ones like r:", ones_like_r)


example1 = torch.tensor([1,2,3]) @ torch.tensor([[4],[3],[2]])
print("Example 1 (vector @ matrix):", example1)

# example2 = torch.tensor([[4],[3],[2]]) @ torch.tensor([1,2,3])
# print("Example 2 (matrix @ vector):", example2)

# example3 = torch.tensor([[1,2,3],[1,2,3],[1,2,3]]) @ torch.tensor([[4,3,2]])
# print("Example 3 (matrix @ matrix):", example3)

example4 = torch.tensor([[1,2,3],[1,2,3],[1,2,3]]) @ torch.tensor([4,3,2])
print("Example 4 (matrix @ column vector):", example4)

print("div:",torch.tensor([1,2,3])@torch.tensor([2,2,2]))

# Creating a manual 4x3 tensor
manual_tensor = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])


print("testing auto transpose:",manual_tensor@manual_tensor.T)