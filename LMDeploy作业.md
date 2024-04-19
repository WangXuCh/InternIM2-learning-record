# LMDeploy作业

### 一、配置LMDeploy的运行环境

- 从InternStudio上克隆已有的环境，并激活

  ```
  studio-conda -t lmdeploy -o pytorch-2.1.2
  conda activate lmdeploy
  ```

  运行之后的结果截图

  ![image-20240419130822537](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20240419130822537.png)

- 运行下面的命令安装LMDeploy

  ```
  pip install lmdeploy[all]==0.3.0
  ```

  安装完成的截图

  ![image-20240419131100382](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419131100382.png)

  使用以下命令打印看看当前环境是否有lmdeploy

  ```
  conda list
  ```

  ![image-20240419131222663](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419131222663.png)

  

### 二、LMDeploy模型对话

- 下载模型

  一般情况下是需要从huggingface或者魔塔社区下载需要的模型，但是InternStudio上已经下载好了常用的预训练模型，我们只需要从share文件夹中拷贝过来或者通过创建软链接的方式实现引用。

  下面的指令展示了使用软链接的方式将需要的模型文件添加到目标文件夹下

  ```
  ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/
  ```

  完成之后会发现/root/路径下多了一个名为internlm2-chat-1_8b的文件夹

  ![image-20240419131644348](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419131644348.png)

- 使用transformer库运行模型

  首先使用如下指令创建一个py文件

  ```
  touch /root/pipeline_transformer.py
  ```

  ![image-20240419131849166](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419131849166.png)

  将以下内容复制进去

  ```
  import torch
  from transformers import AutoTokenizer, AutoModelForCausalLM
  
  tokenizer = AutoTokenizer.from_pretrained("/root/internlm2-chat-1_8b", trust_remote_code=True)
  
  # Set `torch_dtype=torch.float16` to load model in float16, otherwise it will be loaded as float32 and cause OOM Error.
  model = AutoModelForCausalLM.from_pretrained("/root/internlm2-chat-1_8b", torch_dtype=torch.float16, trust_remote_code=True).cuda()
  model = model.eval()
  
  inp = "hello"
  print("[INPUT]", inp)
  response, history = model.chat(tokenizer, inp, history=[])
  print("[OUTPUT]", response)
  
  inp = "please provide three suggestions about time management"
  print("[INPUT]", inp)
  response, history = model.chat(tokenizer, inp, history=history)
  print("[OUTPUT]", response)
  ```

  注意检查红框中的路径是否正确

  ![image-20240419131943584](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419131943584.png)

  然后利用下面的命令在终端运行代码

  ```
  python /root/pipeline_transformer.py
  ```

  运行结果如下

  ![image-20240419132830041](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419132830041.png)

- 使用LMDeploy与模型对话

  ```
  lmdeploy chat [HF格式模型路径/TurboMind格式模型路径]
  ```

  具体的，我使用下面的命令运行

  ```
  lmdeploy chat /root/internlm2-chat-1_8b
  ```

  运行之后进行对话的截图

  ![image-20240419133200854](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419133200854.png)

  ![image-20240419133226308](C:\Users\ASUS\Desktop\lmdepoly\LMDeploy作业.assets\image-20240419133226308.png)













































