# XTuner微调LLM实战

- ### 环境配置

  ```
  # 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 的环境：
  # pytorch    2.0.1   py3.10_cuda11.7_cudnn8.5.0_0
  
  studio-conda xtuner0.1.17
  # 如果你是在其他平台：
  # conda create --name xtuner0.1.17 python=3.10 -y
  
  # 激活环境
  conda activate xtuner0.1.17
  # 进入家目录 （~的意思是 “当前用户的home路径”）
  cd ~
  # 创建版本文件夹并进入，以跟随本教程
  mkdir -p /root/xtuner0117 && cd /root/xtuner0117
  
  # 拉取 0.1.17 的版本源码
  git clone -b v0.1.17  https://github.com/InternLM/xtuner
  # 无法访问github的用户请从 gitee 拉取:
  # git clone -b v0.1.15 https://gitee.com/Internlm/xtuner
  
  # 进入源码目录
  cd /root/xtuner0117/xtuner
  
  # 从源码安装 XTuner
  pip install -e '.[all]'
  ```

![image-20240416134658501](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416134658501.png)



- ### 生成微调数据

  ```
  # 前半部分是创建一个文件夹，后半部分是进入该文件夹。
  mkdir -p /root/ft && cd /root/ft
  
  # 在ft这个文件夹里再创建一个存放数据的data文件夹
  mkdir -p /root/ft/data && cd /root/ft/data
  
  # 创建 `generate_data.py` 文件
  touch /root/ft/data/generate_data.py
  
  # 将以下内容复制进创建的py文件中
  import json
  
  # 设置用户的名字
  name = '不要姜葱蒜大佬'   # 需要修改的地方
  # 设置需要重复添加的数据次数
  n =  10000   # 需要修改的地方
  
  # 初始化OpenAI格式的数据结构
  data = [
      {
          "messages": [
              {
                  "role": "user",
                  "content": "请做一下自我介绍"
              },
              {
                  "role": "assistant",
                  "content": "我是{}的小助手，内在是上海AI实验室书生·浦语的1.8B大模型哦".format(name)
              }
          ]
      }
  ]
  
  # 通过循环，将初始化的对话数据重复添加到data列表中
  for i in range(n):
      data.append(data[0])
  
  # 将data列表中的数据写入到一个名为'personal_assistant.json'的文件中
  with open('personal_assistant.json', 'w', encoding='utf-8') as f:
      # 使用json.dump方法将数据以JSON格式写入文件
      # ensure_ascii=False 确保中文字符正常显示
      # indent=4 使得文件内容格式化，便于阅读
      json.dump(data, f, ensure_ascii=False, indent=4)
  ```

![image-20240417165145488](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417165145488.png)

输入以下指令生成数据

```
python /root/ft/data/generate_data.py
```

![image-20240417175739219](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417175739219.png)



- ### 模型准备

  使用 InternLM 最新推出的小模型 InterLM2-Chat-1.8B 来完成此次的微调。

  在 InternStudio 直接通过以下代码一键创建文件夹并将所有文件复制进去

  ```
  # 创建目标文件夹，确保它存在。
  # -p选项意味着如果上级目录不存在也会一并创建，且如果目标文件夹已存在则不会报错。
  mkdir -p /root/ft/model
  
  # 复制内容到目标文件夹。-r选项表示递归复制整个文件夹。
  cp -r /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b/* /root/ft/model/
  ```

  或者使用软链接

  ```
  # 删除/root/ft/model目录
  rm -rf /root/ft/model
  
  # 创建符号链接
  ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-1_8b /root/ft/model
  ```







![image-20240416134841501](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416134841501.png)



- ### 配置文件选择

  XTuner 提供多个开箱即用的配置文件，用户可以通过下列命令查看：

  ```
  # 列出所有内置配置文件
  # xtuner list-cfg
  
  # 假如我们想找到 internlm2-1.8b 模型里支持的配置文件
  xtuner list-cfg -p internlm2_1_8b
  ```

![image-20240416134940486](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416134940486.png)



虽然我们用的数据集并不是 `alpaca` 而是我们自己通过脚本制作的小助手数据集 ，但是由于我们是通过 `QLoRA` 的方式对 `internlm2-chat-1.8b` 进行微调。而最相近的配置文件应该就是 `internlm2_1_8b_qlora_alpaca_e3` ，因此我们可以选择拷贝这个配置文件到当前目录：

```
# 创建一个存放 config 文件的文件夹
mkdir -p /root/ft/config

# 使用 XTuner 中的 copy-cfg 功能将 config 文件复制到指定的位置
xtuner copy-cfg internlm2_1_8b_qlora_alpaca_e3 /root/ft/config
```

![image-20240416135037333](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416135037333.png)



打印文件树

![image-20240416135148980](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240416135148980.png)



- ### 修改配置文件

  先要更换模型的路径以及数据集的路径为我们本地的路径。

  ```
  # 修改模型地址（在第27行的位置）
  - pretrained_model_name_or_path = 'internlm/internlm2-1_8b'
  + pretrained_model_name_or_path = '/root/ft/model'
  
  # 修改数据集地址为本地的json文件地址（在第31行的位置）
  - alpaca_en_path = 'tatsu-lab/alpaca'
  + alpaca_en_path = '/root/ft/data/personal_assistant.json'
  
  
  ```

  还可以对一些重要的参数进行调整，包括学习率（lr）、训练的轮数（max_epochs）等等。由于我们这次只是一个简单的让模型知道自己的身份弟位，因此我们的训练轮数以及单条数据最大的 Token 数（max_length）都可以不用那么大

  ```
  # 修改max_length来降低显存的消耗（在第33行的位置）
  - max_length = 2048
  + max_length = 1024
  
  # 减少训练的轮数（在第44行的位置）
  - max_epochs = 3
  + max_epochs = 2
  
  # 增加保存权重文件的总数（在第54行的位置）
  - save_total_limit = 2
  + save_total_limit = 3
  ```

  为了训练过程中能够实时观察到模型的变化情况，XTuner 也是贴心的推出了一个 `evaluation_inputs` 的参数来让我们能够设置多个问题来确保模型在训练过程中的变化是朝着我们想要的方向前进的。比如说我们这里是希望在问出 “请你介绍一下你自己” 或者说 “你是谁” 的时候，模型能够给你的回复是 “我是XXX的小助手...” 这样的回复。因此我们也可以根据这个需求进行更改

  ```
  # 修改每多少轮进行一次评估（在第57行的位置）
  - evaluation_freq = 500
  + evaluation_freq = 300
  
  # 修改具体评估的问题（在第59到61行的位置）
  # 可以自由拓展其他问题
  - evaluation_inputs = ['请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai']
  + evaluation_inputs = ['请你介绍一下你自己', '你是谁', '你是我的小助手吗']
  ```

  这样修改完后在评估过程中就会显示在当前的权重文件下模型对这几个问题的回复了。

  由于我们的数据集不再是原本的 aplaca 数据集，因此我们也要进入 PART 3 的部分对相关的内容进行修改。包括说我们数据集输入的不是一个文件夹而是一个单纯的 json 文件以及我们的数据集格式要求改为我们最通用的 OpenAI 数据集格式。

  ```
  # 把 OpenAI 格式的 map_fn 载入进来（在第15行的位置）
  - from xtuner.dataset.map_fns import alpaca_map_fn, template_map_fn_factory
  + from xtuner.dataset.map_fns import openai_map_fn, template_map_fn_factory
  
  # 将原本是 alpaca 的地址改为是 json 文件的地址（在第102行的位置）
  - dataset=dict(type=load_dataset, path=alpaca_en_path),
  + dataset=dict(type=load_dataset, path='json', data_files=dict(train=alpaca_en_path)),
  
  # 将 dataset_map_fn 改为通用的 OpenAI 数据集格式（在第105行的位置）
  - dataset_map_fn=alpaca_map_fn,
  + dataset_map_fn=openai_map_fn,
  ```



- ### 模型训练

  - 常规训练

    ```
    # 指定保存路径
    xtuner train /root/ft/config/internlm2_1_8b_qlora_alpaca_e3_copy.py --work-dir /root/ft/train
    ```

​			训练过程截图



![image-20240417165221684](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417165221684.png)

![image-20240417165252617](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417165252617.png)

可以看出大约使用了九分钟时间

- deepspeed加速训练

![image-20240417170851263](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417170851263.png)

![image-20240417170916192](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417170916192.png)

可以看出只用了不到八分钟的时间，不使用加速的情况下九分钟，训练速度有所提升，但不多！



- ### 模型转换

模型转换的本质其实就是将原本使用 Pytorch 训练出来的模型权重文件转换为目前通用的 Huggingface 格式文件，那么我们可以通过以下指令来实现一键转换。

```
# 创建一个保存转换后 Huggingface 格式的文件夹
mkdir -p /root/ft/huggingface

# 模型转换
# xtuner convert pth_to_hf ${配置文件地址} ${权重文件地址} ${转换后模型保存地址}
xtuner convert pth_to_hf /root/ft/train/internlm2_1_8b_qlora_alpaca_e3_copy.py /root/ft/train/iter_768.pth /root/ft/huggingface
```

![image-20240417171621135](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417171621135.png)

![image-20240417171634698](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417171634698.png)



![image-20240417151825098](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417151825098.png)



- ### 模型整合

  我们训练的时候使用了LoRA技术，而这种技术微调出来的模型并非完整的原始模型，只是一个在原始模型基础上的额外参数（adapter）。因此我们需要将训练好的额外参数和原始模型组合起来才能使用

  在 XTuner 中提供了一键整合的指令，但是在使用前我们需要准备好三个地址：原模型的地址、训练好的 adapter 层的地址（转为 Huggingface 格式后保存的部分）以及最终保存的地址。

```
mkdir -p /root/ft/final_model

# 解决一下线程冲突的 Bug 
export MKL_SERVICE_FORCE_INTEL=1

# 进行模型整合
# xtuner convert merge  ${NAME_OR_PATH_TO_LLM} ${NAME_OR_PATH_TO_ADAPTER} ${SAVE_PATH} 
xtuner convert merge /root/ft/model /root/ft/huggingface /root/ft/final_model
```

![image-20240417152956518](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417152956518.png)

​		整合完成后，在保存位置可以看到以下内容：

![image-20240418085938006](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418085938006.png)

- ### 对话测试

  XTuner 中也直接的提供了一套基于 transformers 的对话代码，让我们可以直接在终端与 Huggingface 格式的模型进行对话操作。我们只需要准备我们刚刚转换好的模型路径并选择对应的提示词模版（prompt-template）即可进行对话。假如 prompt-template 选择有误，很有可能导致模型无法正确的进行回复。

  ```
  # 与模型进行对话
  xtuner chat /root/ft/final_model --prompt-template internlm2_chat
  ```

![image-20240417175313926](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240417175313926.png)

- ### Web端部署测试

  首先，安装web端需要的依赖库

  ```
  pip install streamlit==1.24.0
  ```

  然后下载InternLM项目代码

  ```
  # 创建存放 InternLM 文件的代码
  mkdir -p /root/ft/web_demo && cd /root/ft/web_demo
  
  # 拉取 InternLM 源文件
  git clone https://github.com/InternLM/InternLM.git
  
  # 进入该库中
  cd /root/ft/web_demo/InternLM
  ```

  将`/root/ft/web_demo/InternLM/chat/web_demo.py` 中的内容替换为以下内容

  ```
  """This script refers to the dialogue example of streamlit, the interactive
  generation code of chatglm2 and transformers.
  
  We mainly modified part of the code logic to adapt to the
  generation of our model.
  Please refer to these links below for more information:
      1. streamlit chat example:
          https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
      2. chatglm2:
          https://github.com/THUDM/ChatGLM2-6B
      3. transformers:
          https://github.com/huggingface/transformers
  Please run with the command `streamlit run path/to/web_demo.py
      --server.address=0.0.0.0 --server.port 7860`.
  Using `python path/to/web_demo.py` may cause unknown problems.
  """
  # isort: skip_file
  import copy
  import warnings
  from dataclasses import asdict, dataclass
  from typing import Callable, List, Optional
  
  import streamlit as st
  import torch
  from torch import nn
  from transformers.generation.utils import (LogitsProcessorList,
                                             StoppingCriteriaList)
  from transformers.utils import logging
  
  from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip
  
  logger = logging.get_logger(__name__)
  
  
  @dataclass
  class GenerationConfig:
      # this config is used for chat to provide more diversity
      max_length: int = 2048
      top_p: float = 0.75
      temperature: float = 0.1
      do_sample: bool = True
      repetition_penalty: float = 1.000
  
  
  @torch.inference_mode()
  def generate_interactive(
      model,
      tokenizer,
      prompt,
      generation_config: Optional[GenerationConfig] = None,
      logits_processor: Optional[LogitsProcessorList] = None,
      stopping_criteria: Optional[StoppingCriteriaList] = None,
      prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                  List[int]]] = None,
      additional_eos_token_id: Optional[int] = None,
      **kwargs,
  ):
      inputs = tokenizer([prompt], padding=True, return_tensors='pt')
      input_length = len(inputs['input_ids'][0])
      for k, v in inputs.items():
          inputs[k] = v.cuda()
      input_ids = inputs['input_ids']
      _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
      if generation_config is None:
          generation_config = model.generation_config
      generation_config = copy.deepcopy(generation_config)
      model_kwargs = generation_config.update(**kwargs)
      bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
          generation_config.bos_token_id,
          generation_config.eos_token_id,
      )
      if isinstance(eos_token_id, int):
          eos_token_id = [eos_token_id]
      if additional_eos_token_id is not None:
          eos_token_id.append(additional_eos_token_id)
      has_default_max_length = kwargs.get(
          'max_length') is None and generation_config.max_length is not None
      if has_default_max_length and generation_config.max_new_tokens is None:
          warnings.warn(
              f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                  to control the generation length. "
              'This behaviour is deprecated and will be removed from the \
                  config in v5 of Transformers -- we'
              ' recommend using `max_new_tokens` to control the maximum \
                  length of the generation.',
              UserWarning,
          )
      elif generation_config.max_new_tokens is not None:
          generation_config.max_length = generation_config.max_new_tokens + \
              input_ids_seq_length
          if not has_default_max_length:
              logger.warn(  # pylint: disable=W4902
                  f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                  f"and 'max_length'(={generation_config.max_length}) seem to "
                  "have been set. 'max_new_tokens' will take precedence. "
                  'Please refer to the documentation for more information. '
                  '(https://huggingface.co/docs/transformers/main/'
                  'en/main_classes/text_generation)',
                  UserWarning,
              )
  
      if input_ids_seq_length >= generation_config.max_length:
          input_ids_string = 'input_ids'
          logger.warning(
              f"Input length of {input_ids_string} is {input_ids_seq_length}, "
              f"but 'max_length' is set to {generation_config.max_length}. "
              'This can lead to unexpected behavior. You should consider'
              " increasing 'max_new_tokens'.")
  
      # 2. Set generation parameters if not already defined
      logits_processor = logits_processor if logits_processor is not None \
          else LogitsProcessorList()
      stopping_criteria = stopping_criteria if stopping_criteria is not None \
          else StoppingCriteriaList()
  
      logits_processor = model._get_logits_processor(
          generation_config=generation_config,
          input_ids_seq_length=input_ids_seq_length,
          encoder_input_ids=input_ids,
          prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
          logits_processor=logits_processor,
      )
  
      stopping_criteria = model._get_stopping_criteria(
          generation_config=generation_config,
          stopping_criteria=stopping_criteria)
      logits_warper = model._get_logits_warper(generation_config)
  
      unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
      scores = None
      while True:
          model_inputs = model.prepare_inputs_for_generation(
              input_ids, **model_kwargs)
          # forward pass to get next token
          outputs = model(
              **model_inputs,
              return_dict=True,
              output_attentions=False,
              output_hidden_states=False,
          )
  
          next_token_logits = outputs.logits[:, -1, :]
  
          # pre-process distribution
          next_token_scores = logits_processor(input_ids, next_token_logits)
          next_token_scores = logits_warper(input_ids, next_token_scores)
  
          # sample
          probs = nn.functional.softmax(next_token_scores, dim=-1)
          if generation_config.do_sample:
              next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
          else:
              next_tokens = torch.argmax(probs, dim=-1)
  
          # update generated ids, model inputs, and length for next step
          input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
          model_kwargs = model._update_model_kwargs_for_generation(
              outputs, model_kwargs, is_encoder_decoder=False)
          unfinished_sequences = unfinished_sequences.mul(
              (min(next_tokens != i for i in eos_token_id)).long())
  
          output_token_ids = input_ids[0].cpu().tolist()
          output_token_ids = output_token_ids[input_length:]
          for each_eos_token_id in eos_token_id:
              if output_token_ids[-1] == each_eos_token_id:
                  output_token_ids = output_token_ids[:-1]
          response = tokenizer.decode(output_token_ids)
  
          yield response
          # stop when each sentence is finished
          # or if we exceed the maximum length
          if unfinished_sequences.max() == 0 or stopping_criteria(
                  input_ids, scores):
              break
  
  
  def on_btn_click():
      del st.session_state.messages
  
  
  @st.cache_resource
  def load_model():
      model = (AutoModelForCausalLM.from_pretrained('/root/ft/final_model',
                                                    trust_remote_code=True).to(
                                                        torch.bfloat16).cuda())
      tokenizer = AutoTokenizer.from_pretrained('/root/ft/final_model',
                                                trust_remote_code=True)
      return model, tokenizer
  
  
  def prepare_generation_config():
      with st.sidebar:
          max_length = st.slider('Max Length',
                                 min_value=8,
                                 max_value=32768,
                                 value=2048)
          top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
          temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
          st.button('Clear Chat History', on_click=on_btn_click)
  
      generation_config = GenerationConfig(max_length=max_length,
                                           top_p=top_p,
                                           temperature=temperature)
  
      return generation_config
  
  
  user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
  robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
  cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
      <|im_start|>assistant\n'
  
  
  def combine_history(prompt):
      messages = st.session_state.messages
      meta_instruction = ('')
      total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
      for message in messages:
          cur_content = message['content']
          if message['role'] == 'user':
              cur_prompt = user_prompt.format(user=cur_content)
          elif message['role'] == 'robot':
              cur_prompt = robot_prompt.format(robot=cur_content)
          else:
              raise RuntimeError
          total_prompt += cur_prompt
      total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
      return total_prompt
  
  
  def main():
      # torch.cuda.empty_cache()
      print('load model begin.')
      model, tokenizer = load_model()
      print('load model end.')
  
  
      st.title('InternLM2-Chat-1.8B')
  
      generation_config = prepare_generation_config()
  
      # Initialize chat history
      if 'messages' not in st.session_state:
          st.session_state.messages = []
  
      # Display chat messages from history on app rerun
      for message in st.session_state.messages:
          with st.chat_message(message['role'], avatar=message.get('avatar')):
              st.markdown(message['content'])
  
      # Accept user input
      if prompt := st.chat_input('What is up?'):
          # Display user message in chat message container
          with st.chat_message('user'):
              st.markdown(prompt)
          real_prompt = combine_history(prompt)
          # Add user message to chat history
          st.session_state.messages.append({
              'role': 'user',
              'content': prompt,
          })
  
          with st.chat_message('robot'):
              message_placeholder = st.empty()
              for cur_response in generate_interactive(
                      model=model,
                      tokenizer=tokenizer,
                      prompt=real_prompt,
                      additional_eos_token_id=92542,
                      **asdict(generation_config),
              ):
                  # Display robot response in chat message container
                  message_placeholder.markdown(cur_response + '▌')
              message_placeholder.markdown(cur_response)
          # Add robot response to chat history
          st.session_state.messages.append({
              'role': 'robot',
              'content': cur_response,  # pylint: disable=undefined-loop-variable
          })
          torch.cuda.empty_cache()
  
  
  if __name__ == '__main__':
      main()
  ```

​		同样，在本地的PowerShell中ssh远程连接当前开发机，需要密码的时候复制开发机的密码

```
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的端口号>
```

​		然后在开发机运行

```
streamlit run /root/ft/web_demo/InternLM/chat/web_demo.py --server.address 127.0.0.1 --server.port 6006
```

![image-20240418091025001](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418091025001.png)

打开[http://127.0.0.1:6006](http://127.0.0.1:6006/)后会加载模型，加载完成之后即可开始对话

![image-20240418091157364](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418091157364.png)

严重过拟合，发啥都只回复“我是千里的小助手内在1.8B大模型书生·浦语的1.8B大模型哦”！！！！



# 使用XTuner微调自己的模型

按照以上步骤构建自己的数据（我这里使用的是外科的医疗问答数据），模型选择InternLM-7B，同时准备配置文件，模型预训练权重。

- ### 数据构建



![image-20240418104735279](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418104735279.png)

![image-20240418104643296](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418104643296.png)



- ### 微调

使用所有数据微调需要一天左右的时间，训练五千条数据的话，微调只需要五十分钟左右

![image-20240418105657140](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418105657140.png)

- ### 网页demo



![image-20240418133945661](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418133945661.png)

![image-20240418133956676](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418133956676.png)

![image-20240418134031949](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418134031949.png)

![image-20240418134040519](https://github.com/WangXuCh/InternIM2-learning-record/blob/main/typora-user-images/image-20240418134040519.png)

可以看出来回答的很详细也很准确。自己实践了一圈下来，Xtuner这个架构确实是很方便，用户基本上只需要构建自己的对话数据集就可以了，剩下的全部交给XTuner。
