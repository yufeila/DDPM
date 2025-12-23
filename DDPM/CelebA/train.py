import torch
import os
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.unet import SimpleUNet
from dataset import CelebAHQDataset, get_transforms
from utils.diffusion_utils import *

# --- 配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./CelebA-HQ/raw/" 
save_dir = "./checkpoints_celeb256_third"
os.makedirs(save_dir, exist_ok=True)

# 断点续训配置
load_path = "/4T/whf/qt/DDPM/myddpm/checkpoints_celeb256_third/ddpm_step100000.pt"
num_steps = 200000  
step_counter = 100000

# --- 初始化 ---
dataset = CelebAHQDataset(data_path, transform=get_transforms())
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 加载权重
if os.path.exists(load_path):
    model.load_state_dict(torch.load(load_path, map_location=device))
    print(f"Loaded checkpoint: {load_path}")

def train_step(x):
    batch_size = x.shape[0]
    t = torch.randint(0, T, (batch_size,), device=device).long()
    noise = torch.randn_like(x)
    
    # 扩散加噪
    x_noisy = extract(sqrt_alphas_cumprod, t, x.shape) * x + \
              extract(sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
    
    optimizer.zero_grad()
    noise_pred = model(x_noisy, t)
    loss = F.mse_loss(noise, noise_pred)
    loss.backward()
    optimizer.step()
    return loss.item()

# --- 训练循环 ---
model.train()
stop_training = False

print(f"Starting training from step {step_counter}...")
for epoch in range(1000):
    for x in dataloader:
        x = x.to(device)
        loss = train_step(x)
        step_counter += 1

        if step_counter % 250 == 0:
            print(f"Step {step_counter}, loss: {loss:.4f}")

        if step_counter % 2000 == 0:
            # 保存
            save_path = os.path.join(save_dir, f"ddpm_step{step_counter}.pt")
            torch.save(model.state_dict(), save_path)
            
            # 采样预览
            model.eval()
            samples = sample(model, (4, 3, 256, 256), device)
            samples = torch.clamp((samples + 1) / 2, 0, 1).cpu()
            grid = torchvision.utils.make_grid(samples, nrow=2)
            plt.imshow(grid.permute(1, 2, 0))
            plt.show()
            model.train()

        if step_counter >= num_steps:
            stop_training = True
            break
    if stop_training:
        break
