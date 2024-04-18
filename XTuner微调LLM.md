# XTuner微调LLM：1.8B、多模态和Agent

## 一、Fintune简介

#### 1. 为什么要微调？

利用通用数据集训练出来的LLM通常在某些专业度要求较高的特定领域性能较差，因此可以利用该领域的专业知识对基座LLM进行微调，从而得到在该领域表现较好的LLM。

#### 2.两种Finetune范式：

- ##### 增量预训练微调：

  - 使用场景：让基座模型学习到一些新知识，如某个垂类领域的常识
  - 训练数据：文章、书籍、代码等等

- ##### 指令跟随微调：

  - 使用场景：让模型学会对话模板，根据人类指令进行对话
  - 训练数据：高质量的对话、问答数据

![image-20240416123759714](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416123759714.png)

![image-20240416123908015](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416123908015.png)

#### 3.数据格式

![image-20240416124135699](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416124135699.png)

![image-20240416124219872](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416124219872.png)

![image-20240416124315193](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416124315193.png)

#### 4.LoRA和QLoRA

- #### LoRA总结

##### 之前的fine-tune的方法

- ##### Adapters

  方法：在模型的每一层之间添加可训练的小规模的网络，冻结原始网络权重，以此来减少fine-tune所需要的参数量。

  应用：适用于那些希望在保持预训练模型结构不变的同时，对模型进行特定任务调整的场景。

  缺点：引入推理延时

- ##### Prefix Tuning

  方法：在模型输入部分添加一些可训练的前缀向量，然后将这些向量和数据一起送入模型，改变模型对单独数据的推理结果。

  应用：适用于需要对模型进行轻量级微调的场景，特别是当模型非常大，而可用于训练的资源有限时。

  缺点：鲁棒性不够好，模型的结果严重依赖于前缀的质量（举一个不是很恰当的例子就是：网络本身就没这些只是，你非得加前缀让他说，这怎么能说出来？）

###### 简单来说LoRA就是通过引入两个低秩参数化更新矩阵来减少参数量，我的理解是把参数量降维（变少）

- ##### 问题描述：

  假设一个网络的所有参数W，维度是d * k，微调它的梯度∆W维度也是是d * k，也就是说W和∆W的参数量是一样的，这就给我们训练参数量太大的网络带来困难。同时，如果有不同的下游任务，则需要对每个下游任务都训练出一个这样的∆W，因此这种方式的fine-tune是非常昂贵的。

- ##### 解决方案：

  ![image-20240416124640277](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416124640277.png)

  针对这个问题，文章提出将∆W进行低秩分解，分解成两个矩阵A（维度是d * r）、B（维度是r * k），其中r远远小于d和k的最小值，然后我们就可以计算∆W和AB的参数量：

```
∆W形状是dxk，则参数量=dxk
A的形状是dxr，B的形状是rxk，则A的参数量=dxr，B的参数量=rxd，总参数量=dxr+rxk=rx(d + k)
```

​	需要注意的一点就是，r越小，fine-turn的参数量越少，速度越快，但伴随的是精度的降低，反之速度变慢，精度上升。感觉就是用精度换速度的一种方式。

- ##### 应用：

  需要对大模型所有参数进行微调，但不显著增加计算量的场景

- ##### 优点：

  训练成本降低，训练速度提升，针对不同任务只需训练针对不同任务的AB即可

- ##### 缺点：

  以精度换速度



- #### QLoRA总结


在LoRA的基础上，添加了NF4的数据压缩（信息理论中最有的正太分布数据量化数据类型），进一步减少了显存和内存的消耗；然后添加一组可学习的LoRA权重，这些权重通过量化权重的反向传播梯度进行调整。

![image-20240416124745395](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416124745395.png)

块状 k-bit 量化：既压缩了数据，又解决了异常值（我理解为噪声）对数据压缩的影响。我理解为：数据分布不是线性的，因此利用块量化（类似分治？）进行数据压缩。

- ##### 优点：

  使用NF4量化预训练权重，减少内存。*计算梯度的时候再反量化？量化和反量化的或称会不会带来时间消耗？*

  双重量化：虽然NF4的数据的内存消耗很小，但是将量化常数也占用了内存。~~比如：32位常量，块大小为64，则量化常量每个参数占用32/64=0.5位。因此对量化常量再次量化（线性量化）。第二次量化如果用8位常量，块大小位256，~~

  分页优化：防止爆显存

## 二、XTuner框架

-  傻瓜式：以配置文件的形式封装了大部分微调场景，0基础的非专业人员也能一键开始微调
- 轻量级：对7B量级的LLM，微调所需的最小现存仅为8GB

#### 1.功能亮点

- 适配多种生态：
  - 多种微调算法覆盖各类SFT场景
  - 适配多种开原生态，支持加载Huggingface、ModelScope模型或数据集
  - 自动优化加速，开发者无需关注复杂的显存优化与计算加速细节
- 适配多种硬件
  - 训练方案覆盖NVIDIA 20系列以上所有显卡
  - 最低只需要8GB显存即可微调7B模型

![image-20240416125256725](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416125256725.png)

#### 2.XTuner快速上手

- 安装

  ```
  pip install xtuner
  ```

- 挑选配置模板

  ```
  xtuner list-cfg -p internlm_20b
  ```

![image-20240416125538832](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416125538832.png)



- 一键训练

  ```
  xtuner train internlm_20b_qlora_oasst1_512_e3
  ```

  ![image-20240416125545457](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416125545457.png)

![image-20240416125610072](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416125610072.png)

- 对话

  ```
  xtuner chat internlm/internlm-chat-20b                         # float 16 模型对话
  xtuner chat internlm/internlm-chat-20b --bits 4     	  	   # 4bit 模型对话
  xtuner chat internlm/internlm-chat-20b --adapter $ADAPTER_DIR  # 加载Adapter模型对话
  ```

#### 3.XTuner数据引擎

![image-20240416125935370](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416125935370.png)

![image-20240416130025223](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416130025223.png)

![image-20240416130034959](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416130034959.png)

```
xtuner copy-cfg internlm_20b_qlora_alpaca_e3 ./  # 拷贝配置模板
vi internlm_20b_qlora_alpaca_e3    				 # 修改配置模板
xtuner train internlm_20b_qlora_alpaca_e3  		 # 启动训练
```

## 三、8GB玩转LMM

![image-20240416131803094](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416131803094.png)

![image-20240416132218143](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416132218143.png)

## 四、InternLM2 1.8B 模型

![image-20240416131946796](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416131946796.png)

## 五、多模态LLM微调

#### 1.给LLM装上电子眼：多模态LLM原理简介

![image-20240416132314247](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416132314247.png)

#### 2.什么型号的电子眼：LLaVA方案简介

![image-20240416132426586](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416132426586.png)

#### 3.快速上手

![image-20240416132509611](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416132509611.png)

## 六、Agent

#### 待续。。。
