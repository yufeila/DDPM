import torch
import torch.nn.functional as F

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def sample_step(model, x_t, t, device):
    # 此处为从模型预测噪声并还原图像的一步
    beta_t = extract(betas, t, x_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_t.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x_t.shape)
    
    model_mean = sqrt_recip_alphas_t * (x_t - beta_t * model(x_t, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = extract(posterior_variance, t, x_t.shape)
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample(model, shape, device):
    # 从纯噪声开始生成完整图像
    img = torch.randn(shape, device=device)
    for i in reversed(range(0, T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = sample_step(model, img, t, device)
    return img
