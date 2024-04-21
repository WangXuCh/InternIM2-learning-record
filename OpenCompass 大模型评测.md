# OpenCompass 大模型评测

## 一、OpenCompass介绍

### 1. 为什么要研究大模型的评测

- 首先，研究评测对于我们全面了解大型语言模型的优势和限制至关重要。尽管许多研究表明大型语言模型在多个通用任务上已经达到或超越了人类水平，但仍然存在质疑，即这些模型的能力是否只是对训练数据的记忆而非真正的理解。例如，即使只提供LeetCode题目编号而不提供具体信息，大型语言模型也能够正确输出答案，这暗示着训练数据可能存在污染现象。
- 其次，研究评测有助于指导和改进人类与大型语言模型之间的协同交互。考虑到大型语言模型的最终服务对象是人类，为了更好地设计人机交互的新范式，我们有必要全面评估模型的各项能力。
- 最后，研究评测可以帮助我们更好地规划大型语言模型未来的发展，并预防未知和潜在的风险。随着大型语言模型的不断演进，其能力也在不断增强。通过合理科学的评测机制，我们能够从进化的角度评估模型的能力，并提前预测潜在的风险，这是至关重要的研究内容。
- 对于大多数人来说，大型语言模型可能似乎与他们无关，因为训练这样的模型成本较高。然而，就像飞机的制造一样，尽管成本高昂，但一旦制造完成，大家使用的机会就会非常频繁。因此，了解不同语言模型之间的性能、舒适性和安全性，能够帮助人们更好地选择适合的模型，这对于研究人员和产品开发者而言同样具有重要意义。

### 2. 如何通过能力评测促进模型发展？

- 面向未来拓展能力维度：评测体系需增加新能力维度如数学、复杂推理、逻辑推理、代码和智能体等，以全面评估模型性能。
- 扎根通用能力聚焦垂直行业：在医疗、金融、法律等专业领域，评测需结合行业知识和规范，以评估模型的行业适用性。
- 高质量中文基准：针对中文场景，需要开发能准确评估其能力的中文评测基准，促进中文社区的大模型发展。
- 性能评测反哺能力迭代：通过深入分析评测性能，探索模型能力形成机制，发现模型不足，研究针对性提升策略。

### 3.大语言模型评测中的挑战

- 全面性：
  - 大模型应用场景千变万化
  - 模型能力演进迅速
  - 如何设计和构造可扩展的能力维度体系
- 数据污染：
  - 海量语料不可避免带来评测集污染
  - 需要可靠的数据污染检测技术
  - 如何设计可动态更新的高质量评测基准
- 评测成本：
  - 评测数十万道题需要大量算力资源
  - 基于人工打分的主观评测成本高昂
- 鲁棒性：
  - 大模型对提示词十分敏感
  - 多次采样情况下模型性能不稳定

### 4. OpenCompass 2.0 司南大模型评测体系开源历程

- 2023.05.01 -> 完成Alpha版本开发，支持干亿参数语言大模型高效评测
- 2023.07.06 -> OpenCompass 正式开源，学术评测支持最完善的评测工具之一，支持5大能力维度，70个数据集，40万评测题目
- 2023.08.18 -> OpenCompass 数据和性能对比上线，支持100+开源模型的多维度性能对比
- 2023.09.07 -> 支持多编程语言代码评镜像，发布稳定可复现代码评 测镜像，提供多编程语言能力分析和对比
- 2023.10.26 -> 联合南京大学推出大模型司法能力评测基准，勾建多层能力体系助力法律场景能力分析
- 2023.12.01 -> 发布多模态评测工具套件VLMEvalKit，支持包括Gemini、GPT-4V等商业模型评测支持
- 2024.01.30 -> OpenCompass 2.0司南大模型评测体系正式发布

### 5.评测对象

主要评测对象为语言大模型与多模态大模型。我们以语言大模型为例介绍评测的具体模型类型

- 基座模型：一般是经过海量的文本数据以自监督学习的方式进行训练获得的模型（如OpenAI的GPT-3，Meta的LLaMA），往往具有强大的文字续写能力。
- 对话模型：一般是在的基座模型的基础上，经过指令微调或人类偏好对齐获得的模型（如OpenAI的ChatGPT、上海人工智能实验室的书生·浦语），能理解人类指令，具有较强的对话能力。

### 6.工具架构

![image-20240420090343011](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420090343011.png)

- 模型层：大模型评测所涉及的主要模型种类，OpenCompass以基座模型和对话模型作为重点评测对象。
- 能力层：OpenCompass从本方案从通用能力和特色能力两个方面来进行评测维度设计。在模型通用能力方面，从语言、知识、理解、推理、安全等多个能力维度进行评测。在特色能力方面，从长文本、代码、工具、知识增强等维度进行评测。
- 方法层：OpenCompass采用客观评测与主观评测两种评测方式。客观评测能便捷地评估模型在具有确定答案（如选择，填空，封闭式问答等）的任务上的能力，主观评测能评估用户对模型回复的真实满意度，OpenCompass采用基于模型辅助的主观评测和基于人类反馈的主观评测两种方式。
- 工具层：OpenCompass提供丰富的功能支持自动化地开展大语言模型的高效评测。包括分布式评测技术，提示词工程，对接评测数据库，评测榜单发布，评测报告生成等诸多功能。

### 7. CompassKit 大模型评测全栈工具链

- 数据污染检查：提供多种数据污染检测方法，支持包括GSM-8K.MMLU等主流数据集上的污染检测
- 更丰富的模型推理接入：支持近20个商业模型API支持LMDeploy、vLLM、LighLLM等推理后端
- 长文本能力评测：支持1M长度大海捞针测试，支持多个主流长文本评测基准
- 中英文双语主观评测：支持基于大模型评价的主观评测，提供模型打分、模型对战多种能力，灵活切换上百种评价模型

### 8. 如何评测大模型

OpenCompass采取客观评测与主观评测相结合的方法。针对具有确定性答案的能力维度和场景，通过构造丰富完善的评测集，对模型能力进行综合评价。针对体现模型能力的开放式或半开放式的问题、模型安全问题等，采用主客观相结合的评测方式。

- 客观评测：
  - 客观问答题（生成式评测）：语言翻译、程序生成、逻辑分析等
  - 客观选择题（判别式测评）
- 开放式主观问答：人类专家的主观评测与基于模型打分的主观评测
  - 比如：写一首七言律诗，表达对龙年春节的期待

### 9. OpenCompass评测流水线

![image-20240420091037521](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420091037521.png)

## 二、模型评测实战

- #### 环境配置

  使用如下指令从intern-studio克隆一个现有的环境并激活

  ```
  studio-conda -o internlm-base -t opencompass
  source activate opencompass
  ```

  ![image-20240420092751168](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420092751168.png)

  然后使用git命令下载opencompass，并从源码安装opencompass

  ```
  git clone -b 0.2.4 https://github.com/open-compass/opencompass
  cd opencompass
  pip install -e .
  ```

  ![image-20240420092911534](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420092911534.png)

  如果上述安装未成功，则执行下面的命令，从reauirments.txt中逐一安装依赖

  ```
  pip install -r requirements.txt
  ```

- #### 准备数据

  将评测数据集解压到data/路径下，首先从share文件夹拷贝数据集，然后unzip解压

  ```
  cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
  unzip OpenCompassData-core-20231110.zip
  ```

  ![image-20240420093307826](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420093307826.png)

  

  使用下面的命令查看支持的数据集和模型，可能会出现没有某个库，pip安装就行

  ```
  python tools/list_configs.py internlm ceval
  ```

  ![image-20240420093712045](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420093712045.png)

  ![image-20240420093728118](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240420093728118.png)

- #### 启动评测

  ```
  python run.py --datasets ceval_gen --hf-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-path /share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 1024 --max-out-len 16 --batch-size 2 --num-gpus 1 --debug
  ```

  发现报错

  ![image-20240421110923005](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240421110923005.png)

  ，这个报错看内容说的是MKL_SERVICE_FORCE_INTEL，这不就是XTuner哪一节课实战部分需要解决的线程报错吗？直接运行下面代码

  ```
  export MKL_SERVICE_FORCE_INTEL=1
  ```
  
  之后，再次自动测评，发现报错如下

  ![image-20240421111117713](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240421111117713.png)
  
  按照文档提示，用下面的指令安装protobuf
  
  ```
  pip install protobuf
  ```
  
  ![image-20240421111139089](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240421111139089.png)
  
  然后，再次启动测评
  
  ![image-20240421111834952](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240421111834952.png)
  
  评测完成后显示如下：
  
  ![image-20240421113437745](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240421113437745.png)
  
  ![image-20240421113508943](C:\Users\ASUS\Desktop\Agent\OpenCompass 大模型评测实战.assets\image-20240421113508943.png)
