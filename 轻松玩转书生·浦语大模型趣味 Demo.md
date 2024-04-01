# 轻松玩转书生·浦语大模型趣味 Demo

### <u>基础作业和进阶作业的标题均以三号标题斜体展示出来</u>

## **部署 `InternLM2-Chat-1.8B` 模型进行智能对话**

### Step1：环境配置

1. 创建开发机：镜像选择cuda11.7，算例选10%A100

2. 进入开发机后，在终端输入

   ```
   studio-conda -o internlm-base -t demo
   or
   conda create -n demo python==3.10 -y
   conda activate demo
   conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```

​		创建完成之后如下图：

![image-20240331112154701]([typora-user-images\image-20240331112154701.png](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240331112154701.png))

3. 之后激活对应的环境并按照以下指令安装包

   ```
   pip install huggingface-hub==0.17.3
   pip install transformers==4.34 
   pip install psutil==5.9.8
   pip install accelerate==0.24.1
   pip install streamlit==1.32.2 
   pip install matplotlib==3.8.3 
   pip install modelscope==1.9.5
   pip install sentencepiece==0.1.99
   ```

​		然后可以使用conda list查看是否有刚才安装的包

### Step2：下载**`InternLM2-Chat-1.8B`**模型

1. 按照如下命令创建并进入文件夹（名字也可以紫丁贵，后面引用的时候注意就好）

```
mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo
```

2. 进入demo文件夹，双击打开 `/root/demo/download_mini.py` 文件，复制以下代码：

```python
import os
from modelscope.hub.snapshot_download import snapshot_download

# 创建保存模型目录
os.system("mkdir /root/models")

# save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')
```

3. 使用 `python /root/demo/download_mini.py` 瞎子模型权重。这是modelscope（魔塔社区）自己带的下载脚本，我们也可以根据自己的需要修改 `snapshot_download `中的参数。

​	下载完成后如下所示

![image-20240331113110617](typora-user-images\image-20240331113110617.png)   

### Step3：运行 cli_demo

1. 打开 `/root/demo/cli_demo.py` 文件，复制以下代码：

   ```
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   
   model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
   
   tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
   model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
   model = model.eval()
   
   system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
   - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
   - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
   """
   
   messages = [(system_prompt, '')]
   
   print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")
   
   while True:
       input_text = input("\nUser  >>> ")
       input_text = input_text.replace(' ', '')
       if input_text == "exit":
           break
   
       length = 0
       for response, _ in model.stream_chat(tokenizer, input_text, messages):
           if response is not None:
               print(response[length:], flush=True, end="")
               length = len(response)
   ```

2. 利用如下指令，执行demo

   ```python
   conda activate demo
   python /root/demo/cli_demo.py
   ```

3. 等模型加载完成后，输入对话示例测试

   ```
   请创作一个300字的小故事
   ```

# *基础作业结果展示：*

![image-20240331211124765](typora-user-images\image-20240331211124765.png)

## **实战：部署实战营优秀作品 `八戒-Chat-1.8B` 模型**

### Step1：环境配置

1. 如果已经在demo环境内，则不需要再次激活环境，否则，需要激活环境

2. 利用git命令获取仓库里的demo文件

   ```python
   cd /root/
   git clone https://gitee.com/InternLM/Tutorial -b camp2
   # git clone https://github.com/InternLM/Tutorial -b camp2
   cd /root/Tutorial
   ```

### Step2：下载并运行Chat-八戒Demo

1. 在 `Web IDE` 中执行 `bajie_download.py`：

   ```
   python /root/Tutorial/helloworld/bajie_download.py
   ```

2. 待程序下载完成后，输入运行命令：

   ```
   streamlit run /root/Tutorial/helloworld/bajie_chat.py --server.address 127.0.0.1 --server.port 6006
   ```

​	运行截图：

​	![image-20240331194234776](typora-user-images\image-20240331194234776.png)

3. 待程序运行的同时，对端口环境配置本地 `PowerShell` 。使用快捷键组合 `Windows + R`（Windows 即开始菜单键）打开指令界面，并输入命令，按下回车键。（Mac 用户打开终端即可）。

4. 然后查询当前开发机的端口，根据自己开发机的端口输入命令

   ```
   # 从本地使用 ssh 连接 studio 端口
   # 将下方端口号 38374 替换成自己的端口号
   ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
   ```

5. 复制当前开发机的密码，输入到passwoed中，回车

​	运行结果：

​	![image-20240331201638709](typora-user-images\image-20240331201638709.png)

## **实战：使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型（开启 30% A100 权限后才可开启此章节）**

### **初步介绍 Lagent 相关知识**

Lagent 是一个轻量级、开源的基于大语言模型的智能体（agent）框架，支持用户快速地将一个大语言模型转变为多种类型的智能体，并提供了一些典型工具为大语言模型赋能。它的整个框架图如下:

![image-20240331201723910](typora-user-images\image-20240331201723910.png)

Lagent 的特性总结如下：

- 流式输出：提供 stream_chat 接口作流式输出，本地就能演示酷炫的流式 Demo。
- 接口统一，设计全面升级，提升拓展性，包括：
  - Model : 不论是 OpenAI API, Transformers 还是推理加速框架 LMDeploy 一网打尽，模型切换可以游刃有余；
  - Action: 简单的继承和装饰，即可打造自己个人的工具集，不论 InternLM 还是 GPT 均可适配；
  - Agent：与 Model 的输入接口保持一致，模型到智能体的蜕变只需一步，便捷各种 agent 的探索实现；
- 文档全面升级，API 文档全覆盖。

### Step1:环境配置及相关代码库的下载

1. 打开 `Intern Studio` 界面，调节配置（必须在开发机关闭的条件下进行）：

2. 重新开启开发机，输入命令，开启 conda 环境

   ```
   conda activate demo
   ```

3. cd到文件夹

   ```
   cd /root/demo
   ```

4. 使用 git 命令下载 Lagent 相关的代码库

   ```
   git clone https://gitee.com/internlm/lagent.git
   # git clone https://github.com/internlm/lagent.git
   cd /root/demo/lagent
   git checkout 581d9fb8987a5d9b72bb9ebd37a95efd47d479ac
   pip install -e . # 源码安装
   ```

​	运行效果图

![image-20240331202716221](typora-user-images\image-20240331202716221.png)

### Step2：使用 `Lagent` 运行 `InternLM2-Chat-7B` 模型为内核的智能体

`Intern Studio` 在 share 文件中预留了实践章节所需要的所有基础模型，包括 `InternLM2-Chat-7b` 、`InternLM2-Chat-1.8b` 等等。我们可以在后期任务中使用 `share` 文档中包含的资源，但是在本章节，为了能让大家了解各类平台使用方法，还是推按照提示步骤进行实验。

1. cd到 lagent 路径

2. 在 terminal 中输入指令，构造软链接快捷访问方式：

   ```
   ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b
   ```

3. 打开 `lagent` 路径下 `examples/internlm2_agent_web_demo_hf.py` 文件，并修改对应位置 (71行左右) 代码：

   ```
   # 其他代码...
   value='/root/models/internlm2-chat-7b'
   # 其他代码...
   ```

4. 输入运行命令：

   ```
   streamlit run /root/demo/lagent/examples/internlm2_agent_web_demo_hf.py --server.address 127.0.0.1 --server.port 6006
   ```

5. 然后跟上一个任务一样，配置本地ssh

   ```
   # 从本地使用 ssh 连接 studio 端口
   # 将下方端口号 38374 替换成自己的端口号
   ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 38374
   ```

6. 打开 http://127.0.0.1:6006进行对话（一定勾选数据分析），可以尝试输入以下指令

   ```
   请解方程 2*X=1360 之中 X 的结果
   ```

### 进阶作业：*`Lagent` 工具调用 `数据分析` Demo 部署*

![image-20240331205810686](typora-user-images\image-20240331205810686.png)

![image-20240331210603447](typora-user-images\image-20240331210603447.png)



## 实战：实践部署 `浦语·灵笔2` 模型

#### 初步介绍 `XComposer2` 相关知识

`浦语·灵笔2` 是基于 `书生·浦语2` 大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色，总结起来其具有：

- 自由指令输入的图文写作能力： `浦语·灵笔2` 可以理解自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。
- 准确的图文问题解答能力：`浦语·灵笔2` 具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。
- 杰出的综合能力： `浦语·灵笔2-7B` 基于 `书生·浦语2-7B` 模型，在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 `GPT-4V` 和 `Gemini Pro`。

![image-20240401074516594](typora-user-images\image-20240401074516594.png)

### Step1：环境配置并下载源码（此任务需要50%A100）

1. 启动demo环境，并补充需要的包

   ```
   conda activate demo
   # 补充环境包
   pip install timm==0.4.12 sentencepiece==0.1.99 markdown2==2.4.10 xlsxwriter==3.1.2 gradio==4.13.0 modelscope==1.9.5
   ```

2. 下载 **InternLM-XComposer 仓库** 相关的代码资源：

   ```
   cd /root/demo
   git clone https://gitee.com/internlm/InternLM-XComposer.git
   # git clone https://github.com/internlm/InternLM-XComposer.git
   cd /root/demo/InternLM-XComposer
   git checkout f31220eddca2cf6246ee2ddf8e375a40457ff626
   ```

3. 在 `terminal` 中输入指令，构造软链接快捷访问方式：

   ```
   ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b /root/models/internlm-xcomposer2-7b
   ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm-xcomposer2-vl-7b /root/models/internlm-xcomposer2-vl-7b
   ```

### Step2：图文写作实战

1. 输入指令，用于启动 `InternLM-XComposer`：

   ```
   cd /root/demo/InternLM-XComposer
   python /root/demo/InternLM-XComposer/examples/gradio_demo_composition.py  \
   --code_path /root/models/internlm-xcomposer2-7b \
   --private \
   --num_gpus 1 \
   --port 6006
   ```

2. 参照之前的任务配置本地ssh

3. 打开1.中程序生成的地址，直接点击submit

​	结果图

![image-20240401093345616](typora-user-images\image-20240401093345616.png)

### Step3：图片理解实战

1. 关闭并重新启动一个新的 `terminal`，继续输入指令，启动 `InternLM-XComposer2-vl`

   ```
   conda activate demo
   
   cd /root/demo/InternLM-XComposer
   python /root/demo/InternLM-XComposer/examples/gradio_demo_chat.py  \
   --code_path /root/models/internlm-xcomposer2-vl-7b \
   --private \
   --num_gpus 1 \
   --port 6006
   ```

2. 参照之前的任务配置本地ssh

3. 打开1.中程序生成的地址，上传一张图片，并输入以下句子

   ```
   请分析一下图中内容
   ```

### 	进阶作业：*`浦语·灵笔2` 的 `图文创作` 及 `视觉问答` 部署*

![image-20240401124042727](typora-user-images\image-20240401124042727.png)

### *进阶作业：使用huggingface下载包，下载`InternLM2-Chat-7B` 的 `config.json` 文件到本地*

1. 使用 `Hugging Face` 官方提供的 `huggingface-cli` 命令行工具。安装依赖:

   ```
   pip install -U huggingface_hub
   ```

2. 新建python文件，输入以下代码，其中resume-download代表断点续下，local-dir代表本地存储路径

   ```python
   import os
   # 下载模型
   os.system('huggingface-cli download --resume-download internlm/internlm2-chat-7b --local-dir your_path')
   ```

​	![image-20240401125100191](typora-user-images\image-20240401125100191.png)

3. 在终端运行python文件

   ![image-20240401125241972](typora-user-images\image-20240401125241972.png)

​		![image-20240401125607609](typora-user-images\image-20240401125607609.png)

3. 以下内容将展示使用 `huggingface_hub` 下载模型中的部分文件

   ```python
   import os 
   from huggingface_hub import hf_hub_download  # Load model directly 
   
   hf_hub_download(repo_id="internlm/internlm2-7b", filename="config.json")
   ```

![image-20240401130257122](typora-user-images\image-20240401130257122.png)

