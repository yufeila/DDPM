"""
Basic functionality test for diffusion_torch package.

Run this script to verify the migration is working correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 50)
    print("Testing imports...")
    
    try:
        from diffusion_torch import nn as diffusion_nn
        print("  ✓ diffusion_torch.nn")
        
        from diffusion_torch import utils
        print("  ✓ diffusion_torch.utils")
        
        from diffusion_torch import diffusion_utils
        print("  ✓ diffusion_torch.diffusion_utils")
        
        from diffusion_torch.models import UNet
        print("  ✓ diffusion_torch.models.UNet")
        
        from diffusion_torch.data_utils import (
            get_dataset, normalize_data, unnormalize_data,
            setup_distributed, is_main_process,
            DistributedSummaryWriter, ScalarTracker,
        )
        print("  ✓ diffusion_torch.data_utils")
        
        print("\n✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_nn_layers():
    """Test neural network layers."""
    print("\n" + "=" * 50)
    print("Testing nn layers...")
    
    from diffusion_torch import nn as diffusion_nn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Test get_timestep_embedding
    t = torch.randint(0, 1000, (4,), device=device)
    emb = diffusion_nn.get_timestep_embedding(t, 128)
    assert emb.shape == (4, 128), f"Expected (4, 128), got {emb.shape}"
    print(f"  ✓ get_timestep_embedding: {emb.shape}")
    
    # Test Dense layer
    dense = diffusion_nn.Dense(128, 256).to(device)
    out = dense(emb)
    assert out.shape == (4, 256), f"Expected (4, 256), got {out.shape}"
    print(f"  ✓ Dense: {out.shape}")
    
    # Test Conv2d layer (NCHW)
    conv = diffusion_nn.Conv2d(3, 64, filter_size=3).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    out = conv(x)
    assert out.shape == (4, 64, 32, 32), f"Expected (4, 64, 32, 32), got {out.shape}"
    print(f"  ✓ Conv2d: {out.shape}")
    
    # Test NIN layer
    nin = diffusion_nn.NIN(64, 128).to(device)
    out = nin(out)
    assert out.shape == (4, 128, 32, 32), f"Expected (4, 128, 32, 32), got {out.shape}"
    print(f"  ✓ NIN: {out.shape}")
    
    # Test helper functions
    flat = diffusion_nn.meanflat(out)
    assert flat.shape == (4,), f"Expected (4,), got {flat.shape}"
    print(f"  ✓ meanflat: {flat.shape}")
    
    print("\n✅ All nn layers work correctly!")
    return True


def test_unet():
    """Test UNet model."""
    print("\n" + "=" * 50)
    print("Testing UNet model...")
    
    from diffusion_torch.models import UNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create small UNet for testing
    model = UNet(
        in_ch=3,
        ch=64,  # Small for testing
        out_ch=3,
        num_res_blocks=1,
        attn_resolutions=(8,),  # Attention at 8x8
        dropout=0.0,
        ch_mult=(1, 2, 2),
        
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    
    with torch.no_grad():
        out = model(x, t)
    
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  ✓ Forward pass: input {x.shape} -> output {out.shape}")
    
    print("\n✅ UNet model works correctly!")
    return True


def test_diffusion():
    """Test GaussianDiffusion."""
    print("\n" + "=" * 50)
    print("Testing GaussianDiffusion...")
    
    from diffusion_torch.diffusion_utils import GaussianDiffusion, get_beta_schedule
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create beta schedule
    betas = get_beta_schedule(
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02,
        num_diffusion_timesteps=1000,
    )
    print(f"  ✓ Beta schedule: shape={betas.shape}, range=[{betas.min():.4f}, {betas.max():.4f}]")
    
    # Create diffusion
    diffusion = GaussianDiffusion(betas=betas, loss_type='noisepred', device=device)
    print(f"  ✓ GaussianDiffusion created, T={diffusion.num_timesteps}")
    
    # Test q_sample (forward diffusion)
    x_start = torch.randn(4, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (4,), device=device)
    
    x_t = diffusion.q_sample(x_start, t)
    assert x_t.shape == x_start.shape
    print(f"  ✓ q_sample: {x_start.shape} -> {x_t.shape}")
    
    # Test q_mean_variance
    mean, var, log_var = diffusion.q_mean_variance(x_start, t)
    assert mean.shape == x_start.shape
    print(f"  ✓ q_mean_variance: mean={mean.shape}")
    
    # Test predict_start_from_noise
    noise = torch.randn_like(x_start)
    x_pred = diffusion.predict_start_from_noise(x_t, t, noise)
    assert x_pred.shape == x_start.shape
    print(f"  ✓ predict_start_from_noise: {x_pred.shape}")
    
    # Test q_posterior
    post_mean, post_var, post_log_var = diffusion.q_posterior(x_start, x_t, t)
    assert post_mean.shape == x_start.shape
    print(f"  ✓ q_posterior: mean={post_mean.shape}")
    
    print("\n✅ GaussianDiffusion works correctly!")
    return True


def test_training_step():
    """Test a single training step."""
    print("\n" + "=" * 50)
    print("Testing training step...")
    
    from diffusion_torch.models import UNet
    from diffusion_torch.diffusion_utils import GaussianDiffusion, get_beta_schedule
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create small model
    model = UNet(
        in_ch=3,
        ch=32,
        out_ch=3,
        num_res_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        ch_mult=(1, 2),
    ).to(device)
    
    # Create diffusion
    betas = get_beta_schedule('linear', beta_start=0.0001, beta_end=0.02, 
                              num_diffusion_timesteps=1000)
    diffusion = GaussianDiffusion(betas=betas, loss_type='noisepred', device=device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training step
    model.train()
    x_start = torch.randn(2, 3, 32, 32, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    
    def denoise_fn(x, t):
        return model(x, t)
    
    losses = diffusion.p_losses(denoise_fn, x_start, t)
    loss = losses.mean()
    
    print(f"  ✓ Loss computed: {loss.item():.4f}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  ✓ Backward pass and optimizer step completed")
    
    print("\n✅ Training step works correctly!")
    return True


def test_sampling():
    """Test sampling (just a few steps for speed)."""
    print("\n" + "=" * 50)
    print("Testing sampling (short run)...")
    
    from diffusion_torch.models import UNet
    from diffusion_torch.diffusion_utils import GaussianDiffusion, get_beta_schedule
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create very small model
    model = UNet(
        in_ch=3,
        ch=32,
        out_ch=3,
        num_res_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        ch_mult=(1,),
    ).to(device)
    model.eval()
    
    # Use only 10 timesteps for fast testing
    betas = get_beta_schedule('linear', beta_start=0.0001, beta_end=0.02, 
                              num_diffusion_timesteps=10)
    diffusion = GaussianDiffusion(betas=betas, loss_type='noisepred', device=device)
    
    def denoise_fn(x, t):
        return model(x, t)
    
    # Generate 1 sample
    with torch.no_grad():
        samples = diffusion.p_sample_loop(denoise_fn=denoise_fn, shape=(1, 3, 16, 16))
    
    assert samples.shape == (1, 3, 16, 16)
    print(f"  ✓ Generated samples: {samples.shape}")
    print(f"  ✓ Sample range: [{samples.min().item():.2f}, {samples.max().item():.2f}]")
    
    print("\n✅ Sampling works correctly!")
    return True


def test_data_utils():
    """Test data utilities."""
    print("\n" + "=" * 50)
    print("Testing data utilities...")
    
    from diffusion_torch.data_utils import normalize_data, unnormalize_data
    
    # Test normalization
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.int32)
    x_norm = normalize_data(x)
    
    assert x_norm.min() >= -1.0 and x_norm.max() <= 1.0
    print(f"  ✓ normalize_data: [0,255] -> [{x_norm.min().item():.2f}, {x_norm.max().item():.2f}]")
    
    x_back = unnormalize_data(x_norm)
    assert x_back.min() >= 0 and x_back.max() <= 255
    print(f"  ✓ unnormalize_data: [-1,1] -> [{x_back.min().item():.2f}, {x_back.max().item():.2f}]")
    
    # Test other utilities
    from diffusion_torch.data_utils import is_main_process, get_rank, get_world_size
    
    print(f"  ✓ is_main_process: {is_main_process()}")
    print(f"  ✓ get_rank: {get_rank()}")
    print(f"  ✓ get_world_size: {get_world_size()}")
    
    print("\n✅ Data utilities work correctly!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("DIFFUSION_TORCH MIGRATION TEST")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    all_passed = True
    
    tests = [
        test_imports,
        test_nn_layers,
        test_unet,
        test_diffusion,
        test_training_step,
        test_sampling,
        test_data_utils,
    ]
    
    for test_fn in tests:
        try:
            passed = test_fn()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("The migration is working correctly.")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("=" * 50 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
