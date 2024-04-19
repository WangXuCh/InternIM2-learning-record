## Lagent & AgentLego 智能体应用搭建实战

#### 1. 准备开发环境

- 创建开发机，选择cuda12.2镜像，选择30%A100

- 从intern-studio克隆一个已有的环境

  ```
  studio-conda -t agent -o pytorch-2.1.2
  ```

- 创建一个用于存放Agent文件的目录

  ```
  mkdir -p /root/agent
  ```

  环境配置完的截图

  ![image-20240419174832884](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419174832884.png)

#### 2. 安装Lagent和AgentLego

```
cd /root/agent
conda activate agent
git clone https://gitee.com/internlm/lagent.git
cd lagent && git checkout 581d9fb && pip install -e . && cd ..
git clone https://gitee.com/internlm/agentlego.git
cd agentlego && git checkout 7769e0d && pip install -e . && cd ..
```

![image-20240419174948003](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419174948003.png)

![image-20240419175205489](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419175205489.png)

![image-20240419175311705](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419175311705.png)

#### 3. 安装其他依赖

```
conda activate agent
pip install lmdeploy==0.3.0
```

![image-20240419175608942](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419175608942.png)

![image-20240419175655411](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419175655411.png)

#### 4. 准备Tutorial

```
cd /root/agent
git clone -b camp2 https://gitee.com/internlm/Tutorial.git
```

<img src="https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419175719065.png" alt="image-20240419175719065" style="zoom:200%;" />



#### 5. Lagent：轻量级智能体框架

- #### Lagent Web Demo

  - 由于 Lagent 的 Web Demo 需要用到 LMDeploy 所启动的 api_server，因此我们首先按照下图指示在 vscode terminal 中执行如下代码使用 LMDeploy 启动一个 api_server。

    ```
    conda activate agent
    lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b \
                                --server-name 127.0.0.1 \
                                --model-name internlm2-chat-7b \
                                --cache-max-entry-count 0.1
    ```

    ![image-20240419180303727](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419180303727.png)

  - 新建一个终端窗口，启动并使用 Lagent Web Demo

    ```
    cd /root/agent/lagent/examples
    streamlit run internlm2_agent_web_demo.py --server.address 127.0.0.1 --server.port 7860
    ```

    ![image-20240419180749690](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419180749690.png)

  - 同样的，在本地的终端里面，ssh连接当前开发机

    ```
    ssh -CNg -L 7860:127.0.0.1:7860 -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p 你的 ssh 端口号
    ```

  - 接着打开网址 [http://localhost:7860](http://localhost:7860/)，并天机啊模型IP、选择插件，如下图所示

    ![image-20240419180948738](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419180948738.png)

  - 接着，就可以输入prompt进行论文的检索了，比如输入如下prompt

    ```
    请帮我搜索 InternLM2 Technical Report
    ```

    ![image-20240419181123294](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419181123294.png)

    ```
    请帮我搜索与Depth Estimation有关的论文
    ```

    ![image-20240419181244145](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419181244145.png)



#### 6. 直接使用AgentLego

- 首先下载demo文件

  ```
  cd /root/agent
  wget http://download.openmmlab.com/agentlego/road.jpg
  ```

  ![image-20240419205143647](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419205143647.png)

- 安装AgentLego所需要的额外依赖库

  ```
  conda activate agent
  pip install openmim==0.3.9
  mim install mmdet==3.3.0
  ```

  ![image-20240419210613856](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419210613856.png)

![image-20240419210658573](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419210658573.png)

![image-20240419210730784](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419210730784.png)

- 利用下面的指令创建py文件

  ```
  touch /root/agent/direct_use.py
  ```

- 将下面的内容复制进py文件里

  ```
  import re
  
  import cv2
  from agentlego.apis import load_tool
  
  # load tool
  tool = load_tool('ObjectDetection', device='cuda')
  
  # apply tool
  visualization = tool('/root/agent/road.jpg')
  print(visualization)
  
  # visualize
  image = cv2.imread('/root/agent/road.jpg')
  
  preds = visualization.split('\n')
  pattern = r'(\w+) \((\d+), (\d+), (\d+), (\d+)\), score (\d+)'
  
  for pred in preds:
      name, x1, y1, x2, y2, score = re.match(pattern, pred).groups()
      x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), int(score)
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
      cv2.putText(image, f'{name} {score}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
  
  cv2.imwrite('/root/agent/road_detection_direct.jpg', image)
  ```

- 执行代码

  ```
  python /root/agent/direct_use.py
  ```

  ![image-20240419211618457](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419211618457.png)

  ![image-20240419211714530](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419211714530.png)

  ![image-20240419211922057](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240419211922057.png)
