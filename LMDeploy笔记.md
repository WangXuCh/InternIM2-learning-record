# LMDeploy笔记

## 一、大模型部署的背景

- #### 模型部署

  - ##### 定义：

    - 在软件工程中，部署通常指的是将开发完毕的软件投入使用的过程。
    - 在人工智能领域，模型部署是实现深度学习算法落地应用的关键步骤。简单来说，模型部署就是将训练好的深度学习模型在特定环境中运行的过程。

  - ##### 场景：

    - 服务器端:CPU部署，单GPU/TPU/NPU部署，多卡/集群部署等等
    - 移动端/边缘端:移动机器人，手机等等

![image-20240419134042248](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419134042248.png)



- #### 大模型部署面临的挑战

  - ##### 计算量极大

    - 大模型参数量巨大，前向推理时需要进行大量计算。
    - 根据InternLM2技术报告!1!提供的模型参数数据，以及OpenAl团队提供的计算量估算方法，20B模型每生成1个token，就要进行约406亿次浮点运算;照此计算，若生成128个token，就要进行5.2万亿次运算。
    - 20B算是大模型里的“小”模型了，若模型参数规模达到175B(GPT-3)，Batch-Size(BS)再大一点，每次推理计算量将达到干万亿量级。
    - 以NVIDIA A100为例，单张理论FP16运算性能为每秒77.97 TFLOPS3](77万亿)，性能捉紧。

    ![image-20240419134005312](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419134005312.png)

  - ##### 内存开销巨大

    - 以FP16为例，20B模型仅加载参数就需40G+显存:175B模型(如GPT-3)更是需要350G+显存。
    - 大模型在推理过程中，为避免重复计算，会将计算注意力(Attention)得到的KV进行缓存。根据InternLM2技术报告!1提供的模型参数数据，以及KV Cache空间估算方法[2)，以FP16为例，在batch-size为16、输入512 tokens、输出32 tokens的情境下，仅20B模型就会产生10.3GB的缓存。
    - 目前，以NVIDIA RTX 4060消费级显卡为例(参考零售价￥2399B))，单卡显存仅有8GB;NVIDIA A100单卡显存仅有80GB.

    ![image-20240419134239699](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419134239699.png)

  - ##### 访存瓶颈

    - 大模型推理是“访存密集”型任务。目前硬件计算速度“远快于”显存带宽，存在严重的访存性能瓶颈。
    - 以RTX 4090推理175B大模型为例，BS为1时计算量为6.83TFLOPS，远低于82.58 TFLOPS的FP16计算能力:但访存量为32.62 TB，是显存带宽每秒处理能力的30倍。

    ![image-20240419134359967](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419134359967.png)

  - ##### 动态请求

    - 请求量不确定；
    - 请求时间不确定；
    - Token逐个生成，生成数量不确定。



## 二、大模型部署的方法

- #### 模型剪枝（Pruning）

  剪枝指移除模型中不必要或多余的组件，比如参数，以使模型更加高效。通过对模型中贡献有限的冗余参数进行剪枝，在保证性能最低下降的同时，可以减小存储需求、提高计算效率。

  - ##### 非结构化剪枝  SparseGPT，LoRAPrune，Wanda

    指移除个别参数，而不考虑整体网络结构。这种方法通过将低于阈值的参数置零的方式对个别权重或神经元进行处理。

  - ##### 结构化剪枝  LLM-Pruner

    根据预定义规则移除连接或分层结构，同时保持整体网络结构。这种方法一次性地针对整组权重，优势在于降低模型复杂性和内存使用，同时保持整体的LLM结构完整。

  ![image-20240419134706286](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419134706286.png)

- #### 知识蒸馏（Knowledge Distillation，KD）

  知识蒸馏是一种经典的模型压缩方法，核心思想是通过引导轻量化的学生模型“模仿”性能更好、结构更复杂的教师模型，在不改变学生模型结构的情况下提高其性能。

  - ##### 上下文学习（ICL）：ICL distillation

  - ##### 思维链（CoT）：MT-COT，Fine-tune-CoT等 

  - ##### 指令跟随（IF）：LaMini-LM

  ![image-20240419134955190](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419134955190.png)

- #### 量化（Quantization）

  量化技术将传统的表示方法中的浮点数转换为整数或其他离散形式，以减轻深度学习模型的存储和计算负担。

  - ##### 量化感知训练(QAT)  LLM-QAT

    量化目标无缝地集成到模型的训练过程中。这种方法使LLM在训练过程中适应低精度表示。

  - ##### 量化感知微调(QAF)  PEQA，QLORA

    QAF涉及在微调过程中对LLM进行量化。主要目标是确保经过微调的LLM在量化为较低位宽后仍保持性能。

  - ##### 训练后量化(PTQ)  LLM.int8，AWQ

    在LLM的训练阶段完成后对其参数进行量化。PTQ的主要目标是减少LLM的存储和计算复杂性，而无需对LLM架构进行修改或进行重新训练。

  ![image-20240419135227835](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419135227835.png)



## 三、LMDeploy简介

#### LMDeploy 由 MMDeploy 和 MMRazor 团队联合开发是涵盖了 LLM 任务的全套轻量化、部署和服务解决方案。核心功能包括高效推理、可靠量化、便捷服务和有状态推理。

- #### 优势

  - ##### 高效的推理:LMDeploy开发了Continuous Batch，Blocked K/ Cache，动态拆分和融合，张量并行，高效的计算kernel等重要特性。InternLM2推理性能是vLLM的 1.8 倍。

  - ##### 可靠的量化:LMDeploy支持权重量化和k/v量化。4bit模型推理效率是FP16下的2.4倍。量化模型的可靠性已通过OpenCompass评测得到充分验证。

  - ##### 便捷的服务:通过请求分发服务，LMDeploy 支持多模型在多机、多卡上的推理服务。

  - ##### 有状态推理:通过缓存多轮对话过程中Attention的k/，记住对话历史，从而避免重复处理历史会话。显著提升长文本多轮对话场景中的效率。

- #### 核心功能

  - ##### 模型高效推理

    ```
    命令 -> lmdeploy chat -h
    ```

    TurboMind是LMDeploy团队开发的一款关于 LLM 推理的高效推理引擎。它的主要功能包括:LLaMa 结构模型的支持continuous batch推理模式和可扩展的 KV 缓存管理器。

  - ##### 模型量化压缩

    ```
    命令 -> lmdeploy lite-h
    ```

    W4A16量化(AWQ):将FP16的模型权重量化为INT4，内核计算时，访存量直接降为FP16模型的1/4，大幅降低了访存成本。权重是指仅量化权重，数值计算依然采用FP16(需要将INT4权重反量化)。

- #### 性能表现

  ##### LMDeploy TurboMind 引擎拥有卓越的推理能力，在各种规模的模型上，每秒处理的请求数是 VLLM的1.36~1.85 倍。在静态推理能力方面，TurboMind 4bit 模型推理速度(out token/s)远高于FP16/BF16推理。在小batch时，提高到2.4倍。

  ![image-20240419135731103](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419135731103.png)

- #### 推理视觉多模态大模型

  LMDeploy提供了对视觉多模态大模型llava的支持，有方便的运行代码

  ![image-20240419135913177](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419135913177.png)

  ![image-20240419135921341](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419135921341.png)

- #### 支持的模型

  ![image-20240419135956179](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy笔记.assets\image-20240419135956179.png)

