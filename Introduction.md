# Introduction

## 要求

​	针对一个算法设计问题，完成算法设计、性能与复杂度分析和编程，提交完整的代码和报告。使用的编程语言不限，但是算法部分不能用库函数实现，同时算法不能过于浅显/简单。某些模块的代码如果参考了他人成果，需要明确指出出处，不能抄袭。比如，可以实现一个Alpha Go、对本学院相关专业某篇高质量论文中的算法进行复现（需要搭建完整的仿真环境并给出仿真结果）、利用深度学习、强化学习、遗传算法、动态规划等技术完成某个任务或者功能。



## 前置知识: U-net







## GAN

1. Videos
   1. 入门级必看：https://www.youtube.com/watch?v=TpMIssRdhco&t=335s
      1. 浅显易懂，不涉及任何数学原理
      2. 介绍GAN的典型应用
2. Blogs:
   1. 还未看：https://lilianweng.github.io/posts/2017-08-20-gan/
3. Papers:

## VAE

1. Videos
   1. 必看：https://www.youtube.com/watch?v=9zKuYvjFFS8
      1. 简单介绍数学原理
      2. 附带tensorflow代码片段解读
      3. 优质网络结构图，助于理解
      4. KL散度：衡量两个分布的接近程度
   2. 
2. Papers



## DDPM

1. Videos:

   1. 必看：B站李宏毅教授对Diffusion model的解析：https://www.bilibili.com/video/BV19EVUzrEF4?spm_id_from=333.788.player.switch&vd_source=32f40349361d0bab1e182b57125838ec&p=6

      1. 数学原理清晰易懂，推导过程详细。

      2. 建议从第4个视频:Diffusion原理剖析: 1_4开始看起。

      3. 时间较长：长达1h， 因此，建议1.5倍速/2倍速。

      4. 内容概括：

         1. 4_1: 结合DDPM论文中的算法伪代码，结合ppt演示模型的训练过程和推理过程

            1. forward process(无参): 前向加噪
            2. Reverse process(含参): 后向去噪

         2. 4_2: 

            1. 首先介绍影像生成模型的机制和目标：影像生成模型用来预测真实世界中某一类集合的概率分布，它接受一个从确定性分布(通常是高斯分布)中的采样，通过网络预测出采样输入对应的输出，由于输入是概率分布，输出也是概率分布。**所有影像生成模型本质上的共同目标都是最大化似然函数（等价于最小化KL散度）**
               $$
               input:q(x)\\
               转移概率： p(y|x)\\
               output: w(y) = \int p(y|x)q(x)dx 
               $$

            2. 其次介绍了VAE和DDPM的目标函数。在训练中，由于$\log p_{\theta}(x)$难以计算，最终采用的目标函数是它的lower Bound。从$p_{\theta}(x)$出发，推导出了VAE和DDPM的lower bound， 它们具有相同的形式。
               ![image-20251213112947152](/Users/yufei/course/算法设计与智能计算/project2/DDPM/assets/image-20251213112947152.png)

         3. 4_3: 

            1. 利用随机噪声之间的不不相关性,推导出前向传播的概率分布$q(x_t|x_0)$($t$取任意值)的概率分布.

            2. 对4_2中的Lower Bound进行改写,改写成另一种形式
               ![image-20251213155831351](/Users/yufei/course/算法设计与智能计算/project2/DDPM/assets/image-20251213155831351.png)

            3. 对上式去掉无关项$KL(q(x_T|X_0)||P(x_T))$, 将**重点**放在最小化第3项的KL散度上. 利用贝叶斯公式思想计算出确定分布(期望的分布)$q(x_{t-1}|x_t,x_0)$, 通过假定两分布具有相同的方差,即在概率空间中具有相同的半径,将重点**再次**聚焦到两分布**均值的距离**上.

            4. 期望均值是$\frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t x_0+\sqrt{\alpha}_t (1-\overline{\alpha}_{t-1})x_t}{1-\overline{\alpha_t}}$, 去噪声模型(Denoised model)的输出是$P(x_{t-1}|x_t)$(**反向生成**概率分布)的均值
               ![image-20251213160225523](/Users/yufei/course/算法设计与智能计算/project2/DDPM/assets/image-20251213160225523.png)

            5. 之后利用$x_t$的构造替换掉期望均值的$x_t$, 得到Algorithm 2 Sampling算法中的4:中的逆向传播过程.
               $$
               x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\alpha_t}}\epsilon_{\theta}(x_t,t))+ \sigma _t z
               $$

            6. 4_4:

               1. 介绍了为什么在Algorithm 2 Sampling算法的4中还要加上高斯噪声$\sigma z_t$, 李教授给了一个自己的解读;
               2. 介绍了相关应用: 图像降噪,语音降噪, 文字降噪(这块目前不是很懂,等后续可以深入研究)

   2. 提高：https://www.youtube.com/watch?v=fbLgFrlTnGU
      1. 需要看完VAE下的(a), 作为前置知识
      2. 数学前置知识：贝叶斯公式、马尔可夫链
      3. 优点是在视频中明确标注了公式和原理的来源论文
      4. 具有详细的数学推导，但推导过程较复杂，建议别花太多精力在弄懂公式的每一步上。
      5. 比较详细的阐述了基于DDPM的分类器引导采样(这一块我没看懂)

2. Codes

   1. DDPM: 原始的DDPM的代码使用TensorFlow实现的, 但很遗憾我并不会TensorFlow, 所以我不打算将其作为浮现的第一个代码. 代码仓库见下
      ```
      https://github.com/hojonathanho/diffusion
      
      ```

      

   2. IDDPM: OpenAI 开源的工业界Diffusion Model的架构,在DDPM上做了重要改进, **有PyTorch 版本**, 代码已集成到项目仓库DDPM的DDPM/Improved DDPM/Code/improved-diffusion/improved_diffusion文件夹下.
      ```
      https://github.com/openai/improved-diffusion
      ```






## 进度

(1) 目前,我已跑通IDDPM的代码, 由于按照OpenAI的模型配置会导致参数量过大, 使用单张A800不能在短期内训练完,我对模型参数进行了调整,尤其是U-net的卷积层大小和残差块数.下面是我的参数:
```shell
export MODEL_FLAGS="\
--image_size 32 \
--num_channels 96 \
--num_res_blocks 2 \
--learn_sigma True \
--dropout 0.1"

# 和官方论文中的扩散参数保持一致
export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# 添加了模型保存设置
export TRAIN_FLAGS="\
--lr 1e-4 \
--batch_size 32 \
--use_fp16 True \
--ema_rate 0.9999 \
--log_interval 100 \
--save_interval 10000"

```

OpenAI原始参数:
```shell
# 模型参数：32x32图片，128通道，3个残差块，学习方差
export MODEL_FLAGS="\
--image_size 32 \
--num_channels 128 \
--num_res_blocks 3 \
--learn_sigma True \
--dropout 0.3"

# 扩散参数：4000步扩散，余弦噪声调度
export DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"

# 训练参数：学习率，Batch Size (根据显存调整，如 32 或 64)
export TRAIN_FLAGS="--lr 1e-4 --batch_size 32"
```



(3) 分工: 噪声预测网络(U-net)的训练需要较大的算力,我将在我的服务器上运行, 我已提供了训练出的模型,只需要用`/scripts/image-sample.py`运行相应步长的模型,就可得到.npz文件, 打印前16张图,即可得到反向去燥效果.
