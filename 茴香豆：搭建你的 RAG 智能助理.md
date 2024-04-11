# 茴香豆：搭建你的 RAG 智能助理

LLM对知识库中没有的问题回答时会产生幻觉，为了解决这个问题，通常会将缺失的知识加入训练数据中对LLM重新训练，但这种方式比较贵。因此，有了RAG（检索增强生成）技术。

## RAG技术概览

- RAG是一种结合了检索和生成的技术，旨在通过利用外部知识库来增强大语言模型的性能。它通过检索与用户输入相关的信息片段，并结合这些信息来生成更准确、丰富的回答。

- 优点：能够解决LLM再处理知识密集型任务十可能遇到的困难。RAG能提供更准确的回答、降低训练成本、实现外部记忆。
- 应用：问答系统、文本生成、信息检索、图片描述

## RAG效果对比

如图所示，由于茴香豆是一款比较新的应用， `InternLM2-Chat-7B` 训练数据库中并没有收录到它的相关信息。左图中关于 huixiangdou 的 3 轮问答均未给出准确的答案。右图未对 `InternLM2-Chat-7B` 进行任何增训的情况下，通过 RAG 技术实现的新增知识问答。

![image-20240407101140114](img\image-20240407101140114.png)

### RAG工作原理

- 一般LLM工作原理：用户query -> 回答
- RAG工作原理：外部知识库转换成向量存储在数据库 -> 将用户的query转换成向量并在数据库中查找相关块 -> 将查找到的相关块和问题向量一起作为prompt输入到LLM中，生成最终的回答。

### 向量数据库

- 数据存储：将文本及其他数据通过其他预训练模型转换成固定长度的向量表示，这些向量可以捕捉文本的语义信息/

- 相似性检索：根据用户的查询向量，使用向量数据库快速找到最相关向量的过程。通常通过计算余弦相似度或其他相似度度量来完成。检索结果根据相似度得分进行排序，最相关的文档将被用于后续的文本生成。

  ![image-20240411164316087](img\image-20240411164316087.png)

  ![image-20240411164334881](img\image-20240411164334881.png)

- 向量表示的优化：包括使用更高级的文本编码技，如句子嵌入或段落嵌入，以及对数据库进行优化以支持大规模向量搜索。

### RAG发展进程

RAG的概念最早由Meta的Lewis等人在2020《Retrieval-Augmented Generation for Konowledge-Intensive NLP Tasks》中提出。

![image-20240411164549516](img\image-20240411164549516.png)

Naive RAG: 问答系统、信息检索

Advanced RAG: 摘要生成、内容推荐

Modular RAG: 多模态任务、对话系统

### RAG常见的优化方法

![image-20240411164727939](img\image-20240411164727939.png)

### RAG vs. FineTuning

#### RAG

1. 特点：
   - 非参数记忆，利用外部知识库提供实时更新的信息。
   - 能够处理知识密集型任务，提供准确的事实性回答。
   - 通过检索增强，可以生成更多样化的内容。

2. 使用场景：
   - 适用于需要结合最新信息和实时数据的任务：开放域回答、实时新闻摘要等。

3. 优势：
   - 动态知识更新，处理长尾知识问题。

3. 局限：
   - 依赖于外部知识库的质量和覆盖范围，依赖大模型能力。

 FineTuning

1. 特点：
   - 参数记忆，通过在特定任务数据上训练，模型可以更好的适应该任务。
   - 需要大量标注数据来进行有效微调。
   - 微调后的模型可能过拟合，导致泛化能力下降。

2. 使用场景：
   - 适用于数据可用且需要模型高度专业化的任务，如特定领域的文本分类、情感分析、文本生成等。

3. 优势：
   - 模型性能针对特定任务优化。

3. 局限：
   - 需要大量的数据标注，且对新任务的适应性较差。

### LLM模型优化方法比较

![image-20240411165617203](img\image-20240411165617203.png)

### 评价框架和基准测试

![image-20240411165644793](img\image-20240411165644793.png)





## 茴香豆

茴香豆十一个基于LLM的领域知识助手，由书生浦语团队开发的开源大模型。

- 特点：

  转为即时通讯工具中的群聊场景优化的工作流，提供及时准确的技术支持和自动化问答服务

  通过应用RAG技术，茴香豆能够理解和高效准确的回应与特定知识领域相关的复杂查询

- 应用场景：
  - 智能客服：技术支持、领域知识对话
  - IM工具中创建用户群组，讨论、解答相关问题
  - 随着用户数量的增加，答复内容高度重复，充斥着大量无意义闲聊，人工恢复，成本高，影响工作效率
  - 茴香豆通过提供自动化的问答支持，帮助维护者减轻负担，同时确保用户问题得到有效解答

- 场景难点：
  - 群聊中信息量大，内容多样，从技术讨论到闲聊
  - 用户问题与个人紧密相关，需要准确、实时的专业知识解答
  - 传统NLP解决方案无法准确解析用户意图，且往往无法提供满意答案
  - 需要一个能在群聊中准确识别与回答相关问题的智能助手，同时避免造成消息过载

![image-20240411172637450](img\image-20240411172637450.png)

![image-20240411172845869](img\image-20240411172845869.png)

![image-20240411172858695](img\image-20240411172858695.png)





## 实践环节

### Step1：环境配置

1. 在Intern Studio服务器上创建开发机，选择 `Cuda11.7-conda` 镜像。使用 `30% A100 * 1` ，点击创建并进入开发机。

2. 进入开发机后，从官方环境复制运行 InternLM 的基础环境，命名为 `InternLM2_Huixiangdou`，在命令行模式下运行：

   ```
   studio-conda -o internlm-base -t InternLM2_Huixiangdou
   ```

3. 激活虚拟环境

   ```
   conda activate InternLM2_Huixiangdou
   ```

​	复制并激活环境的示意图![image-20240407102845521](img\image-20240407102845521.png)

4. 下载基础文件

   ```
   # 创建模型文件夹
   cd /root && mkdir models
   
   # 复制BCE模型
   ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
   ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1
   
   # 复制大模型参数（下面的模型，根据作业进度和任务进行**选择一个**就行）
   ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
   ```

   结果截图：

   ![image-20240407103050888](img\image-20240407103050888.png)

5. 利用下面的指令安装茴香豆所需依赖

   ```
   # 安装 python 依赖
   # pip install -r requirements.txt
   
   pip install protobuf==4.25.3 accelerate==0.28.0 aiohttp==3.9.3 auto-gptq==0.7.1 bcembedding==0.1.3 beautifulsoup4==4.8.2 einops==0.7.0 faiss-gpu==1.7.2 langchain==0.1.14 loguru==0.7.2 lxml_html_clean==0.1.0 openai==1.16.1 openpyxl==3.1.2 pandas==2.2.1 pydantic==2.6.4 pymupdf==1.24.1 python-docx==1.1.0 pytoml==0.1.21 readability-lxml==0.8.1 redis==5.0.3 requests==2.31.0 scikit-learn==1.4.1.post1 sentence_transformers==2.2.2 textract==1.6.5 tiktoken==0.6.0 transformers==4.39.3 transformers_stream_generator==0.0.5 unstructured==0.11.2
   
   ## 因为 Intern Studio 不支持对系统文件的永久修改，在 Intern Studio 安装部署的同学不建议安装 Word 依赖，后续的操作和作业不会涉及 Word 解析。
   ## 想要自己尝试解析 Word 文件的同学，uncomment 掉下面这行，安装解析 .doc .docx 必需的依赖
   # apt update && apt -y install python-dev python libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
   ```

​	![image-20240407103204218](img\image-20240407103204218.png)

![image-20240407103716874](img\image-20240407103716874.png)

6. 下载茴香豆仓库

   ```
   cd /root
   # 下载 repo
   git clone https://github.com/internlm/huixiangdou && cd huixiangdou
   git checkout 447c6f7e68a1657fce1c4f7c740ea1700bde0440
   ```

   ![image-20240407103727328](img\image-20240407103727328.png)

### Step2：使用茴香豆搭建RAG助手

1. 修改配置文件。用已下载模型的路径替换 `/root/huixiangdou/config.ini` 文件中的默认模型，需要修改 3 处模型地址，分别是:

- ​	命令行输入下面的命令，修改用于向量数据库和词嵌入的模型

  ```
  sed -i '6s#.*#embedding_model_path = "/root/models/bce-embedding-base_v1"#' /root/huixiangdou/config.ini
  ```

- ​	用于检索的重排序模型

  ```
  sed -i '7s#.*#reranker_model_path = "/root/models/bce-reranker-base_v1"#' /root/huixiangdou/config.ini
  ```

- ​	和本次选用的大模型

  ```
  sed -i '29s#.*#local_llm_path = "/root/models/internlm2-chat-7b"#' /root/huixiangdou/config.ini
  ```

​	修改完成后如下图所示：

![image-20240407103955749](img\image-20240407103955749.png)

2. 创建知识库

   本示例中，使用 **InternLM** 的 **Huixiangdou** 文档作为新增知识数据检索来源，在不重新训练的情况下，打造一个 **Huixiangdou** 技术问答助手。

   首先，下载 **Huixiangdou** 语料：

   ```
   cd /root/huixiangdou && mkdir repodir
   git clone https://github.com/internlm/huixiangdou --depth=1 repodir/huixiangdou
   ```

   ![image-20240407104549705](img\image-20240407104549705.png)

   提取知识库特征，创建向量数据库。数据库向量化的过程应用到了 **LangChain** 的相关模块，默认嵌入和重排序模型调用的网易 **BCE 双语模型**，如果没有在 `config.ini` 文件中指定本地模型路径，茴香豆将自动从 **HuggingFace** 拉取默认模型。

   除了语料知识的向量数据库，茴香豆建立接受和拒答两个向量数据库，用来在检索的过程中更加精确的判断提问的相关性，这两个数据库的来源分别是：

   - 接受问题列表，希望茴香豆助手回答的示例问题
     - 存储在 `huixiangdou/resource/good_questions.json` 中
   - 拒绝问题列表，希望茴香豆助手拒答的示例问题
     - 存储在 `huixiangdou/resource/bad_questions.json` 中
     - 其中多为技术无关的主题或闲聊
     - 如："nihui 是谁", "具体在哪些位置进行修改？", "你是谁？", "1+1"

​		运行下面的命令，增加茴香豆相关的问题到接受问题示例中：

```
cd /root/huixiangdou
mv resource/good_questions.json resource/good_questions_bk.json

echo '[
    "mmpose中怎么调用mmyolo接口",
    "mmpose实现姿态估计后怎么实现行为识别",
    "mmpose执行提取关键点命令不是分为两步吗，一步是目标检测，另一步是关键点提取，我现在目标检测这部分的代码是demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth   现在我想把这个mmdet的checkpoints换位yolo的，那么应该怎么操作",
    "在mmdetection中，如何同时加载两个数据集，两个dataloader",
    "如何将mmdetection2.28.2的retinanet配置文件改为单尺度的呢？",
    "1.MMPose_Tutorial.ipynb、inferencer_demo.py、image_demo.py、bottomup_demo.py、body3d_pose_lifter_demo.py这几个文件和topdown_demo_with_mmdet.py的区别是什么，\n2.我如果要使用mmdet是不是就只能使用topdown_demo_with_mmdet.py文件，",
    "mmpose 测试 map 一直是 0 怎么办？",
    "如何使用mmpose检测人体关键点？",
    "我使用的数据集是labelme标注的，我想知道mmpose的数据集都是什么样式的，全都是单目标的数据集标注，还是里边也有多目标然后进行标注",
    "如何生成openmmpose的c++推理脚本",
    "mmpose",
    "mmpose的目标检测阶段调用的模型，一定要是demo文件夹下的文件吗，有没有其他路径下的文件",
    "mmpose可以实现行为识别吗，如果要实现的话应该怎么做",
    "我在mmyolo的v0.6.0 (15/8/2023)更新日志里看到了他新增了支持基于 MMPose 的 YOLOX-Pose，我现在是不是只需要在mmpose/project/yolox-Pose内做出一些设置就可以，换掉demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py 改用mmyolo来进行目标检测了",
    "mac m1从源码安装的mmpose是x86_64的",
    "想请教一下mmpose有没有提供可以读取外接摄像头，做3d姿态并达到实时的项目呀？",
    "huixiangdou 是什么？",
    "使用科研仪器需要注意什么？",
    "huixiangdou 是什么？",
    "茴香豆 是什么？",
    "茴香豆 能部署到微信吗？",
    "茴香豆 怎么应用到飞书",
    "茴香豆 能部署到微信群吗？",
    "茴香豆 怎么应用到飞书群",
    "huixiangdou 能部署到微信吗？",
    "huixiangdou 怎么应用到飞书",
    "huixiangdou 能部署到微信群吗？",
    "huixiangdou 怎么应用到飞书群",
    "huixiangdou",
    "茴香豆",
    "茴香豆 有哪些应用场景",
    "huixiangdou 有什么用",
    "huixiangdou 的优势有哪些？",
    "茴香豆 已经应用的场景",
    "huixiangdou 已经应用的场景",
    "huixiangdou 怎么安装",
    "茴香豆 怎么安装",
    "茴香豆 最新版本是什么",
    "茴香豆 支持哪些大模型",
    "茴香豆 支持哪些通讯软件",
    "config.ini 文件怎么配置",
    "remote_llm_model 可以填哪些模型?"
]' > /root/huixiangdou/resource/good_questions.json
```

![image-20240407104626077](img\image-20240407104626077.png)		

​		再创建一个测试用的问询列表，用来测试拒答流程是否起效：

```
cd /root/huixiangdou

echo '[
"huixiangdou 是什么？",
"你好，介绍下自己"
]' > ./test_queries.json
```

​		![image-20240407104653926](img\image-20240407104653926.png)

​		在确定好语料来源后，运行下面的命令，创建 RAG 检索过程中使用的向量数据库：

```
# 创建向量数据库存储目录
cd /root/huixiangdou && mkdir workdir 

# 分别向量化知识语料、接受问题和拒绝问题中后保存到 workdir
python3 -m huixiangdou.service.feature_store --sample ./test_queries.json
```

​		向量数据库的创建需要等待一小段时间，过程约占用 1.6G 显存。

![image-20240407112054201](img\image-20240407112054201.png)

​		完成后，**Huixiangdou** 相关的新增知识就以向量数据库的形式存储在 `workdir` 文件夹下。

​		检索过程中，茴香豆会将输入问题与两个列表中的问题在向量空间进行相似性比较，判断该问题是否应该回答，避免群聊过程中的问答泛滥。确定的回答的问题会利用基础模型提取关键词，在知识库中检索 `top K` 相似的 `chunk`，综合问题和检索到的 `chunk` 生成答案。

![image-20240407112127367](img\image-20240407112127367.png)

3. 运行茴香豆知识助手

   ```
   # 填入问题
   sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py
   
   # 运行茴香豆
   cd /root/huixiangdou/
   python3 -m huixiangdou.main --standalone
   ```

   RAG 技术的优势就是非参数化的模型调优，这里使用的仍然是基础模型 `InternLM2-Chat-7B`， 没有任何额外数据的训练。面对同样的问题，我们的**茴香豆技术助理**能够根据我们提供的数据库生成准确的答案：

​	![image-20240407112612425](img\image-20240407112612425.png)

![image-20240407112622986](img\image-20240407112622986.png)

