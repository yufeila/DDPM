# Introduction

## 要求

​	针对一个算法设计问题，完成算法设计、性能与复杂度分析和编程，提交完整的代码和报告。使用的编程语言不限，但是算法部分不能用库函数实现，同时算法不能过于浅显/简单。某些模块的代码如果参考了他人成果，需要明确指出出处，不能抄袭。比如，可以实现一个Alpha Go、对本学院相关专业某篇高质量论文中的算法进行复现（需要搭建完整的仿真环境并给出仿真结果）、利用深度学习、强化学习、遗传算法、动态规划等技术完成某个任务或者功能。

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

            4. 期望均值是$\frac{\sqrt{\overline{\alpha}_{t-1}}\beta_t x_0+\sqrt{\alpha}_t (1-\overline{\alpha}_{t-1})x_t}{1-\overline{\alpha_t}}$, 去噪声模型(Denoised model)的输出是$P(x_{t-1}|x_t)$(反向传播概率分布)的均值
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


