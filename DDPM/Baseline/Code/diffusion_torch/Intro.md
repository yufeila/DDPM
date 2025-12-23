文件夹包含了 DDPM (Denoising Diffusion Probabilistic Models) 的 PyTorch 实现版本。这个代码库是将原始的 TensorFlow 代码迁移到 PyTorch 框架下的结果。

以下是各个文件和模块的详细说明：

1. 核心模块说明
diffusion_utils.py

用途: 包含高斯扩散模型的核心数学逻辑。
主要功能:
get_beta_schedule: 生成扩散过程的 
β
β 调度表（如 linear, cosine 等）。
GaussianDiffusion 类: 实现了扩散过程的前向加噪 (q_sample) 和反向去噪 (p_sample) 逻辑。
normal_kl: 计算两个正态分布之间的 KL 散度。
_warmup_beta: 处理预热阶段的 
β
β 值。
unet.py

用途: 定义用于去噪的 UNet 神经网络架构。
主要功能:
UNet 类: 主模型，包含下采样、中间层和上采样路径。
ResnetBlock: 带有时间步嵌入（Time Embedding）的残差块。
Upsample / Downsample: 上采样和下采样层。
实现了 PyTorch 风格的 forward 前向传播。
nn.py

用途: 提供构建神经网络的基础组件，主要是为了复现 TensorFlow 版本中的特定层行为。
主要功能:
NIN (Network in Network): 相当于 1x1 卷积。
Dense: 全连接层。
default_init: 权重初始化函数，复现了 TF 的 variance_scaling 初始化。
utils.py

用途: 通用工具函数。
主要功能:
SummaryWriter: 封装了 TensorBoard 的日志记录功能。
tile_imgs: 将多张图片拼接成网格以便可视化。
随机种子设置等辅助功能。
2. 数据与脚本模块
data_utils/ 文件夹

datasets.py: 处理数据加载，支持 CIFAR-10 和 CelebA-HQ 等数据集，将数据转换为 PyTorch 的 DataLoader 格式。
dist_utils.py: 分布式训练（DDP）的辅助工具，处理多 GPU 训练时的进程同步。
scripts/ 文件夹

包含针对不同数据集的运行脚本，如 run_cifar.py, run_celebahq.py。
这些脚本是程序的入口点，负责组装模型、数据和训练循环。
3. 如何使用 (Training & Sampling)
以 CIFAR-10 数据集为例 (run_cifar.py)，该脚本支持通过命令行参数控制训练和采样模式。

训练 (Training)
训练模型并将日志/检查点保存到指定目录：

参数说明:
train: 指定模式为训练。
--log_dir: 存放模型权重 (.pt 文件) 和 TensorBoard 日志的路径。
--resume: (可选) 如果需要从中断处继续训练，指定 checkpoint 路径。
采样 (Sampling)
使用训练好的模型生成图像：

参数说明:
sample: 指定模式为采样。
--checkpoint: 必须指定，指向训练好的模型权重文件。
--output_dir: 生成的图片样本保存位置（通常会保存为 .npz 文件和一张预览图 preview.png）。
--num_samples: 要生成的图片总数。
注意: 运行上述命令时，请确保你在 diffusion_torch 的父目录下，或者将

该目录添加到了 PYTHONPATH 中，以便 Python 能正确解析包导入。