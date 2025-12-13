<!--
 * @Author: yyf 17786321727@163.com
 * @Date: 2025-12-12 22:38:44
 * @LastEditors: yyf 17786321727@163.com
 * @LastEditTime: 2025-12-13 10:09:08
 * @FilePath: /DDPM/Introduction.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
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
      1. 数学原理清晰易懂，推导过程详细
      2. 建议从第4个视频:Diffusion原理剖析: 1_4开始看起
      3. 时间较长：长达1h

   2. 提高：https://www.youtube.com/watch?v=fbLgFrlTnGU
      1. 需要看完VAE下的(a), 作为前置知识
      2. 数学前置知识：贝叶斯公式、马尔可夫链
      3. 优点是在视频中明确标注了公式和原理的来源论文
      4. 具有详细的数学推导，但推导过程较复杂，建议别花太多精力在弄懂公式的每一步上。
      5. 比较详细的阐述了基于DDPM的分类器引导采样(这一块我没看懂)


