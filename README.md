
# 🥷 Paper Ninja

## 项目背景

读论文和写论文是科研工作流中不可或缺的过程，正如吴恩达所言—— AI 即是新时代的电力，这场“新能源革命”也早已影响到科研工作流，催生了一批非常优秀的基于 LLM 的文献问答工具，例如 [ChatPaper](https://github.com/kaixindelele/ChatPaper) 、[gpt_academic](https://github.com/binary-husky/gpt_academic) 和 [PaperQA](https://github.com/whitead/paper-qa)。但由于速度，易用性和 APIKEY 等种种原因，笔者对于这些工具都是浅尝辄止，没能有效的将其整合到自己的工作流中来。作为一个练习项目，本项目不打算设计太高的目标，目前计划是结合上述工具与 InternLM2--caht-20B 模型，搭建一个可本地部署的文献总结与文献问答工具，同时巩固训练营所学知识。

## 功能设计

* 文献翻译([gpt_academic](https://github.com/binary-husky/gpt_academic)已有相关插件，配制配置好模型接口即可)
* 文献总结([gpt_academic](https://github.com/binary-husky/gpt_academic)已有相关插件，配制配置好模型接口即可)
* 知识库问答(RAG)
* 数据清洗助手(微调)
* 高效的InternLM2--caht-20B调用接口([gpt_academic](https://github.com/binary-husky/gpt_academic)原生支持InternLM--caht-7b部署，但显存占用极高[RTX4090*2 OOM])

## 实现过程

#### a、Web_UI选型

选择[gpt_academic](https://github.com/binary-husky/gpt_academic)项目做为基础框架，部署详情见原始项目

```bash
micromamba create -n paper_ninja python=3.11.5 -c conda-forge
git clone --depth=1 https://github.com/binary-husky/gpt_academic.git
cd gpt_academic
pip install -r requirements.txt
python main.py

```

#### b、知识库问答

利用[IAnimal](https://ianimal.pro/)知识库中的280万篇摘要构建向量库，检索问题并结合上下文回答问题

```bash
#向量库构建

#编写知识库问答插件

#当前存在的问题
```


#### c、数据清洗助手

利用12703条抗体蛋白信息，生成训练集10041条，测试集2662条，对InternLM2--caht-20B的微调，生成InternLM2--caht-20B-antibody

```bash
#数据集示例

#模型微调

#当前存在的问题
```


#### d、模型量化

实现对Qwen-72B-Chat的**KV Cache 量化以节省显存**

```bash
#lmdeploy部署本地模型(以InternLM2--caht-20B为例)

#配置one-api

#配置gpt_academic兼容one-api

#当前存在的问题
```

#### e、模型部署

结合lmdeploy和One-API完成InternLM2--caht-20B、InternLM2--caht-20B-antibody、Qwen-72B-Chat以及GLM-4的部署

```bash
#lmdeploy部署本地模型(以InternLM2--caht-20B为例)

#配置one-api

#配置gpt_academic兼容one-api

#当前存在的问题
```

#### f、功能测评

根据实际使用场景，主观对比InternLM2--caht-20B、Qwen-72B-Chat以及GLM-4的效果
