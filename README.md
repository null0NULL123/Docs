# Docs

这里不仅是我平时学习和工作中遇到的一些资料和工具的汇总，也是对于知识的总结。 Thanks to 模块是我检索时参考的一些仓库和博客。

## Category

- [Language](#language)
  - [Python](#python)
  - [Golang](#golang)
- [Backend](#backend)
  - [Database](#database)
  - [Middleware](#middleware)
  - [Operator](#operator)
  - [Object Storage](#object-storage)
  - [Server](#server)
  - [Client](#client)
  - [System](#system)
  - [Media](#media)
- [Frontend](#frontend)
- [AI](#ai)
  - [Machine Learning](#ml)
  - [Deep Learning](#dl)
  - [Reinforcement Learning](#rl)
  - [Computer Vision](#computer-vision)
  - [Large Language Models](#llm)
  - [Prompt Engineering](#prompt-engineering)
  - [Retrieval-Augmented Generation](#rag)
    - [Overview](#overview)
    - [Pipeline](#pipeline)
    - [Paper](#paper)
    - [Model](#model)
    - [Project](#project)
  - [Agent](#agent)
  - [AGI](#agi)
- [Tool](#tool)
- [Thanks to](#thanks-to)

---

## Language

[MRO](https://opendylan.org/_static/c3-linearization.pdf)

### Python

- Official
  - [Python Documentation](https://docs.python.org/3/)

  - [PEP](https://peps.python.org/)

  - [cpython](https://github.com/python/cpython)
- Accelerate
  - [Python compilers](https://github.com/cython/cython?tab=readme-ov-file#differences-to-other-python-compilers)
  - [Taichi](https://github.com/taichi-dev/taichi)

### Golang

- [Tour of Go](https://go.dev/tour/list)

## Backend

- [小林 coding](https://xiaolincoding.com/)

### Database

- _SQLite_
- [MySQL](https://dev.mysql.com/doc/refman/8.4/en/)

  - _高性能 MySQL（第 4 版）_

- [Redis](https://redis.io/docs/latest/develop/)

- [neo4j](https://neo4j.com/)
- [ClickHouse](https://clickhouse.com/docs/en/)

### Middleware

- [Kafka](https://kafka.apache.org/documentation/)
- [RocketMQ](https://rocketmq.apache.org/docs/quick-start/)

- [Nginx](https://nginx.org/en/docs/)
- [Traefik](https://doc.traefik.io/traefik/)

- [Nacos](https://nacos.io/zh-cn/docs/what-is-nacos.html)
- [Apollo](https://www.apolloconfig.com/#/zh/README)

- [Elasticsearch](https://github.com/elastic/elasticsearch)

- [gRPC](https://grpc.io/docs/)

### Operator

- [Docker](https://docs.docker.com/)
- [Kubernetes](https://kubernetes.io/docs/home/)
- [Prometheus](https://github.com/prometheus/prometheus)
- [Grafana](https://grafana.com/docs/grafana/latest/)

### Object Storage

- [MinIO](https://docs.min.io/enterprise/aistor-object-store/)
- [SeaweedFS](https://github.com/seaweedfs)

### Server

- Python
  - [fastapi](https://fastapi.tiangolo.com/)
  - [bottle](https://gitlab.com/bottle/bottle)

### Client

- [aiohttp](https://docs.aiohttp.org/en/stable/)

### System

- [Linux](https://www.kernel.org/doc/html/latest/)

- [xv6-riscv](https://github.com/mit-pdos/xv6-riscv)

### Media

- [FFmpeg](https://ffmpeg.org/documentation.html)

## Frontend

### Design To Code

- [imgcook](https://www.imgcook.com/docs/imgcook)

## AI

### ML

- Guide

  - [鸢尾花书](https://github.com/Visualize-ML)
  - [sklearn](https://scikit-learn.org/stable/user_guide.html)

- Algorithm
  - _ARIMA_
  - [LightGBM](https://arxiv.org/abs/1711.07977)

### DL

- Book
  - [Deep Learning](https://github.com/exacity/deeplearningbook-chinese)
  - [D2L](https://zh.d2l.ai/)

- Training & Inference
  - Paper
    - [LoRA](https://arxiv.org/abs/2106.09685)
    - [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
  - Project
    - [ONNX Runtime](https://onnxruntime.ai/)

### RL

- _Proximal Policy Optimization Algorithms_
  - [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
- [RLHF](https://github.com/OpenRLHF/OpenRLHF)

### Computer Vision
- IOU
  - [GIOU](https://arxiv.org/pdf/1902.09630)
- YOLO
  - [ultralytics](https://docs.ultralytics.com/zh/)
  - [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
  - [TOOD: Task-aligned One-stage Object Detection](https://arxiv.org/pdf/2108.07755)
- Annotation Tool
  - [CVAT](https://github.com/cvat-ai/cvat)
  - [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- Cross Domain
  - [DANN](https://arxiv.org/pdf/1505.07818)
  - [CDAN](https://github.com/thuml/CDAN)
  - [CD-FER-Benchmark](https://github.com/HCPLab-SYSU/CD-FER-Benchmark.git)
- MOT
  - [SORT](https://github.com/abewley/sort)
  - _Kalman Filter_
  - _The Hungarian Method for the Assignment Problem_
- Visualization
  - [Netron](https://github.com/lutzroeder/netron)

### LLM

#### Architecture

- _RNN_
- _LSTM_
- [Transformer](https://arxiv.org/abs/1706.03762)
- [BERT](https://arxiv.org/abs/1810.04805)
- [Llama](https://github.com/meta-llama/llama-cookbook)
- [DeepSeek-R1](https://arxiv.org/pdf/2501.12948)

#### Platform

- [Hugging Face](https://huggingface.co/)
- [ModelScope](https://www.modelscope.cn/)

### Prompt Engineering

- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### RAG

#### Overview

- [Introduction](https://www.promptingguide.ai/research/rag.en)
- [paper](https://www.promptingguide.ai/research/rag.en#rag-research-insights)
- [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)

#### Pipeline

- Extract
  - [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
  - [MarkItDown](https://github.com/microsoft/markitdown)
  - [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)
- Chunking
  - Rule-based
    - _nltk_
    - _sliding-window_
    - _TextSplitter_
  - Model-based
    - [SeqModel](https://arxiv.org/pdf/2107.09278)
    - [Dense X Retrieval](https://arxiv.org/pdf/2312.06648)

- Indexing
  - _metadata_
- Embedding
  - _Word2Vec_
  - _GloVe_
  - [bge-m3](https://arxiv.org/abs/2402.03216)

- Vector Store
  - [Chroma](https://docs.trychroma.com/docs/overview/introduction)
  - [milvus](https://milvus.io/docs)
  - [Faiss](https://github.com/facebookresearch/faiss/wiki)
- Query
  - _Hypothetical Questions_
  - [HyDE](https://arxiv.org/pdf/2212.10496)
  - Decomposing
  - Rewriting
  - [Step-Back](https://arxiv.org/pdf/2310.06117)

- Retrieval
  - Micro
    - _TF-IDF_
    - _Best Matching 25_
    - _KD-Tree_

    - [HNSW](https://arxiv.org/pdf/1603.09320)
    - [Multi-Index Hashing](https://www.cs.toronto.edu/~norouzi/research/papers/multi_index_hashing.pdf)
    - [Navigating Spreading-out Graph](https://arxiv.org/pdf/1707.00143)
  - Macro
    - Hybrid Search
    - Recursive Retrieval

- [Rerank](https://docs.zilliz.com/docs/reranking)

  - [RRF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
  - Model
    - [BGE Reranker](https://bge-model.com/tutorial/5_Reranking/5.2.html)
- Generation
  - [Summerization](https://python.langchain.com/docs/tutorials/summarization)
  - Paraphrasing
- Evaluation
  - Project
    - [ragas](https://github.com/explodinggradients/ragas)
    - [evalscope](https://github.com/modelscope/evalscope)

#### Paper

- [RAPTOR](https://arxiv.org/abs/2402.03216)
- [ColBERT](https://github.com/stanford-futuredata/ColBERT)
- [GraphRAG](https://arxiv.org/pdf/2404.16130)
- [LightRAG](https://arxiv.org/pdf/2410.05779)

#### Model

- [Sentence-BERT](https://www.sbert.net/)
- [BGE](https://bge-model.com/)

#### Project

- [RAGFlow](https://github.com/infiniflow/ragflow)

### Agent

- [MCP](https://modelcontextprotocol.io/docs/getting-started/intro)
- [langchain](https://github.com/langchain-ai/langchain)
- [llama_index](https://github.com/run-llama/llama_index)

#### IDE

- [Continue](https://github.com/continuedev/continue)
- [Roo Code](https://github.com/RooCodeInc/Roo-Code)

### AGI

- [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547v2)

## Tool

- Material
  - [pymatgen](https://pymatgen.org/)
- Neuroscience
  - [PsychoPy](https://github.com/psychopy/psychopy)
  - [EEGLAB](https://github.com/sccn/eeglab)

## Thanks to

- List
  - [Top-100-stars](https://github.com/EvanLi/Github-Ranking/blob/master/Top100/Top-100-stars.md)
  - [3Blue1Brown](https://www.3blue1brown.com/)
- Python
  - [Tian Gao](https://github.com/gaogaotiantian)
  - [Yuerer's Blog](https://yuerer.com/categories/Python3/)
- RAG
  - [AGI 掘金](https://agijuejin.feishu.cn/wiki/X3olwSF0fiTdhLkDSyncid8Vngd)
  - [syhya's blog](https://syhya.github.io/zh/)
  - [orrrrz's blog](https://orrrrz.github.io/2025/01/18/rag/multi-vector/)
  - [gzyatcnblogs](https://www.cnblogs.com/gzyatcnblogs)
  - [ting1's blog: RAG分块策略](https://www.cnblogs.com/ting1/p/18598176)

- Agent
  - [Claude Code 逆向工程研究仓库](https://github.com/shareAI-lab/analysis_claude_code)

---

**Navigation**: [To Top](#docs) | [Category](#category)
