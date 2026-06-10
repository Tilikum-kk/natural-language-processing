# 📝 翻译摘要系统 — Translation & Summarization System

> **自然语言处理实践 课程设计**
>
> 学号：2023413304 | 班级：23人工智能一 | 姓名：梁昊

---

## 项目简介

本项目实现了一个基于深度学习的**中文→英文翻译摘要系统**，对比三种不同的翻译+摘要流水线方案，并提供 Flask Web 前端进行交互式演示。

系统以《红楼梦》等中文长文本为输入，输出英文摘要，比较不同流程顺序对最终结果质量的影响。

---

## 三种方案

| 方案 | 流程 | 模型组合 |
|------|------|----------|
| **方案1：先摘要再翻译** | 中文 → 中文摘要 → 英文翻译 | 模型1 (摘要) → 模型2 (翻译) |
| **方案2：先翻译再摘要** | 中文 → 英文翻译 → 英文摘要 | 模型3 (翻译) → 模型4 (摘要) |
| **方案3：直接翻译摘要** | 中文 → 英文摘要 (一步) | 模型5 (端到端) |

### 模型约束

为满足课程设计要求，系统使用 **3 个独立模型**，其中：

| 逻辑编号 | 实际模型 | 架构 | 参数量 | 用途 |
|----------|----------|------|--------|------|
| **模型1 = 模型4** | `mT5_multilingual_XLSum` (mT5-small) | T5 / Span Corruption | ~300M | 多语言摘要 |
| **模型2 = 模型3** | `opus-mt-zh-en` (MarianMT) | MarianMT / Bilingual Translation | ~77M | 中→英翻译 |
| **模型5** | `nllb-200-distilled-600M` (NLLB) | BART / Denoising AE + MT | ~600M | 直接翻译+摘要 |

> 三种模型在架构和预训练目标上均有明显区别，满足课程设计对模型差异化的要求。

---

## 项目结构

```
├── app.py                  # Flask Web 应用主文件 (REST API + 前端路由)
├── models.py               # 深度学习模型加载与推理模块
├── train.py                # 模型训练脚本 (微调三个模型)
├── templates/
│   └── index.html          # Web 前端界面 (单页应用)
├── requirements.txt        # Python 依赖清单
├── 红楼梦.txt               # 《红楼梦》全文测试数据
└── 2023413304自然语言处理实践报告.doc  # 课程设计报告
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **Web 框架** | Flask 3.x |
| **深度学习** | PyTorch, Hugging Face Transformers |
| **模型** | mT5, MarianMT, NLLB (mBART) |
| **前端** | 原生 HTML/CSS/JS (单页应用, 响应式布局) |
| **数据处理** | datasets (Hugging Face), python-docx, csv, json |
| **训练优化** | fp16 混合精度, 梯度累积, Early Stopping |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型缓存 (可选)

在 `models.py` 中修改 Hugging Face 缓存路径：

```python
os.environ["HF_HOME"] = "D:/Python/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像
```

### 3. 启动 Web 应用

```bash
python app.py
```

访问 **http://127.0.0.1:5000** 进入交互界面。

### 4. (可选) 训练模型

```bash
# 训练所有三个模型
python train.py

# 训练指定模型
python train.py --model summarizer
python train.py --model translator

# 限制训练样本数量
python train.py --max_samples 5000
```

微调后的模型保存在 `D:/test_output/` 目录，`models.py` 将自动加载。

---

## API 接口

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/api/upload` | POST | 上传 .txt / .docx 文件 |
| `/api/load_hongloumeng` | POST | 加载《红楼梦》测试段落 |
| `/api/execute/<1|2|3>` | POST | 执行指定方案 |
| `/api/execute_all` | POST | 依次执行三种方案 |
| `/api/current_text` | GET | 获取当前加载的文本 |
| `/api/exit` | GET | 退出系统 |

---

## 使用流程

1. **加载文档** — 上传 .txt/.docx 文件或点击"加载红楼梦"按钮
2. **执行方案** — 可选择执行单个方案或一键执行全部三种方案
3. **对比结果** — 界面同时展示三种方案的英文摘要输出，方便对比质量

系统会从文档中随机提取约 2000 字符的段落进行处理。

---

## 训练数据

| 模型 | 数据集 | 类型 |
|------|--------|------|
| 模型1/4 (摘要) | LCSTS, CLTS + CNN-DailyMail, News Summarization | 中/英文摘要 |
| 模型2/3 (翻译) | WMT ZH-EN | 中→英翻译 |
| 模型5 (直接) | 合成数据 (中文原文 + 翻译模型 → 英文摘要) | 跨语言摘要 |

训练数据存放在 `训练数据/训练数据/` 目录。

---

## 硬件要求

- **推理**: CPU 可运行 (加载 ~1GB 模型), 推荐 GPU 加速
- **训练**: 需要 NVIDIA GPU (≥8GB 显存), 代码已针对 RTX 4060 8GB 优化 (小 batch + 梯度累积 + fp16)

---

## License

本项目为课程设计作业，仅供学习参考。
