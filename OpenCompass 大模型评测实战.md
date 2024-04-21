# OpenCompass 大模型评测实战

## 模型评测实战

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
