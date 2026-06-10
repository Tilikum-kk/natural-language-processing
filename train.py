"""
翻译摘要系统 - 模型训练脚本
============================
在提供的训练数据上微调三个深度学习模型，保存到 D:/test_output/。

模型方案设计（满足课程设计约束）:
  - 模型1 = 模型4 → 同一个多语言摘要模型(mT5-small)，可同时处理中/英文摘要
  - 模型2 = 模型3 → 同一个中→英翻译模型(MarianMT)
  - 模型5 → 直接翻译+摘要模型(mBART-mmt)，架构与模型1/2完全不同

三种模型在架构和预训练目标上均有明显区别:
  | 模型       | 架构      | 预训练目标        | 参数量  |
  |-----------|-----------|-------------------|--------|
  | 模型1/4   | mT5-small | Span Corruption   | ~300M  |
  | 模型2/3   | MarianMT  | Bilingual Trans.  | ~77M   |
  | 模型5     | mBART     | Denoising AE+MT   | ~610M  |

训练策略:
  - RTX 4060 8GB 显存适配: 小 batch + 梯度累积 + fp16
  - 每个模型取适量样本训练 2 epoch
  - 训练完成后保存最佳模型到 D:/test_output/
  - 同时保存训练日志和指标到 D:/test_output/training_results.json

数据集来源:
  模型1/4 (摘要):   LCSTS + CLTS (中文摘要) + CNN-DailyMail + News Summ (英文摘要)
  模型2/3 (翻译):   WMT ZH-EN (中→英翻译)
  模型5 (直接):     由翻译+摘要模型合成 (中文原文 → 英文摘要)
"""

import os
import sys
import json
import csv
import logging
import argparse
import time
from typing import List, Dict, Tuple

# ============================================================================
# Hugging Face 镜像和缓存配置
# ============================================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "D:/Python/huggingface_cache"

# 增大 CSV 字段限制，防止新闻长文本读取报错 (默认131KB，设为100MB)
csv.field_size_limit(100 * 1024 * 1024)

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "训练数据", "训练数据")  # 训练数据实际路径
OUTPUT_DIR = "D:/test_output"                            # 模型和结果保存路径
RESULTS_LOG = os.path.join(OUTPUT_DIR, "training_results.json")  # 训练指标汇总

# 最大训练样本数（根据显存和时间合理调整）
MAX_SAMPLES = 20000

# ============================================================================
# 三个模型的训练配置
# ============================================================================
TRAINING_CONFIGS = {
    # ---- 模型1 = 模型4: 多语言摘要模型 (mT5-small) ----
    # mT5: 基于 T5 架构，使用 Span Corruption 预训练，支持101种语言
    # 同一个模型可同时完成中文摘要和英文摘要任务
    "summarizer": {
        "base_model": "google/mt5-small",
        "output_dir": "summarizer",
        "description": "模型1/4-多语言摘要(mT5-small)",
        "max_input_length": 512,
        "max_target_length": 128,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "epochs": 2,
        "learning_rate": 5e-5,
    },
    # ---- 模型2 = 模型3: 中→英翻译模型 (MarianMT) ----
    # MarianMT: 标准的编码器-解码器Transformer，在OPUS平行语料上训练
    # 专门针对中文→英文翻译任务，架构简洁高效
    "translator": {
        "base_model": "Helsinki-NLP/opus-mt-zh-en",
        "output_dir": "translator",
        "description": "模型2/3-中英翻译(MarianMT)",
        "max_input_length": 512,
        "max_target_length": 512,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "epochs": 2,
        "learning_rate": 5e-5,
    },
    # ---- 模型5: 直接翻译+摘要模型 (mBART-mmt) ----
    # mBART: 基于 BART 架构，使用去噪自编码 + 多语言翻译预训练
    # 架构与 mT5 和 MarianMT 完全不同，使用语言 token 控制输入输出语言
    "direct_model": {
        "base_model": "facebook/mbart-large-50-many-to-many-mmt",
        "output_dir": "direct_model",
        "description": "模型5-直接翻译摘要(mBART-mmt)",
        "max_input_length": 512,
        "max_target_length": 150,
        "batch_size": 2,
        "gradient_accumulation": 8,
        "epochs": 2,
        "learning_rate": 3e-5,
    },
}


# ============================================================================
# 数据加载函数
# ============================================================================

def load_lcsts_data(max_samples: int = None) -> List[Dict]:
    """加载 LCSTS 中文摘要数据 (JSON lines 格式)

    LCSTS: Large Scale Chinese Short Text Summarization Dataset
    格式: {"id": ..., "summary": "...", "content": "..."}
    """
    path = os.path.join(DATA_DIR, "中文摘要", "LCSTSNew", "train.json")
    if not os.path.exists(path):
        logger.warning(f"LCSTS 数据文件不存在: {path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                obj = json.loads(line.strip())
                content = obj.get("content", "").replace(" ", "")
                summary = obj.get("summary", "").replace(" ", "")
                if len(content) > 20 and len(summary) > 5:
                    data.append({"source": content, "target": summary})
            except json.JSONDecodeError:
                continue
    logger.info(f"  LCSTS: 加载 {len(data)} 条有效数据")
    return data


def load_clts_data(max_samples: int = None) -> List[Dict]:
    """加载 CLTS 中文摘要数据 (.src/.tgt 格式)

    CLTS: Chinese Long Text Summarization Dataset
    源文件中的中文字符用空格分隔，需要去除空格
    """
    src_path = os.path.join(DATA_DIR, "中文摘要", "CLTS数据集", "train.src")
    tgt_path = os.path.join(DATA_DIR, "中文摘要", "CLTS数据集", "train.tgt")
    if not os.path.exists(src_path) or not os.path.exists(tgt_path):
        logger.warning(f"CLTS 数据文件不存在")
        return []
    data = []
    with open(src_path, "r", encoding="utf-8") as fs, \
         open(tgt_path, "r", encoding="utf-8") as ft:
        for i, (src, tgt) in enumerate(zip(fs, ft)):
            if max_samples and i >= max_samples:
                break
            src = src.strip().replace(" ", "")
            tgt = tgt.strip().replace(" ", "")
            if len(src) > 20 and len(tgt) > 5:
                data.append({"source": src, "target": tgt})
    logger.info(f"  CLTS: 加载 {len(data)} 条有效数据")
    return data


def load_cnn_dailymail_data(max_samples: int = None) -> List[Dict]:
    """加载 CNN-DailyMail 英文摘要数据

    CNN-DailyMail: 新闻文章及其要点摘要
    格式: CSV, 列: id, article, highlights
    """
    path = os.path.join(DATA_DIR, "英文摘要", "CNN-DailyMail_train", "train.csv")
    if not os.path.exists(path):
        logger.warning(f"CNN-DailyMail 数据文件不存在: {path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            article = row.get("article", "").strip()
            highlights = row.get("highlights", "").strip()
            if len(article) > 50 and len(highlights) > 10:
                data.append({"source": article, "target": highlights})
    logger.info(f"  CNN-DailyMail: 加载 {len(data)} 条有效数据")
    return data


def load_news_summarization_data(max_samples: int = None) -> List[Dict]:
    """加载 News Summarization 英文摘要数据

    格式: CSV, 列: (空), ID, Content, Summary, Dataset
    """
    path = os.path.join(DATA_DIR, "英文摘要", "News Summarization_data", "data.csv")
    if not os.path.exists(path):
        logger.warning(f"News Summarization 数据文件不存在: {path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            content = row.get("Content", "").strip()
            summary = row.get("Summary", "").strip()
            if len(content) > 50 and len(summary) > 10:
                data.append({"source": content, "target": summary})
    logger.info(f"  News Summ: 加载 {len(data)} 条有效数据")
    return data


def load_wmt_translation_data(max_samples: int = None) -> List[Dict]:
    """加载 WMT 中英翻译数据

    格式: CSV, 第一行表头 "0,1"
          后续每行: 序号,中文句子,"英文翻译"
    """
    path = os.path.join(DATA_DIR, "翻译", "wmt_zh_en_training_corpus.csv")
    if not os.path.exists(path):
        logger.warning(f"WMT 数据文件不存在: {path}")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头 "0,1"
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            if len(row) >= 2:
                zh = row[0].strip().replace(" ", "")
                en = row[1].strip().strip('"').strip()
                if len(zh) > 15 and len(en) > 5:
                    data.append({"source": zh, "target": en})
    logger.info(f"  WMT ZH-EN: 加载 {len(data)} 条有效数据")
    return data


# ============================================================================
# Tokenize 函数
# ============================================================================

def tokenize_standard(examples, tokenizer, max_input_length, max_target_length):
    """标准 tokenize: 适用于 mT5, MarianMT"""
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        examples["target"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def tokenize_mbart(examples, tokenizer, max_input_length, max_target_length):
    """mBART tokenize: 需要设置源语言为 zh_CN, 目标语言为 en_XX"""
    tokenizer.src_lang = "zh_CN"
    model_inputs = tokenizer(
        examples["source"],
        max_length=max_input_length,
        truncation=True,
        padding=False,
    )
    # 设置目标语言以正确编码 labels
    tokenizer.src_lang = "en_XX"
    labels = tokenizer(
        examples["target"],
        max_length=max_target_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# ============================================================================
# 单个模型训练函数
# ============================================================================

def train_model(
    model_key: str,
    data: List[Dict],
    use_mbart_preprocess: bool = False,
) -> dict:
    """微调单个模型

    Args:
        model_key: 模型配置键 ("summarizer", "translator", "direct_model")
        data: 训练数据列表 [{"source": ..., "target": ...}, ...]
        use_mbart_preprocess: 是否使用 mBART 特殊 tokenize

    Returns:
        dict: 训练结果指标
    """
    config = TRAINING_CONFIGS[model_key]
    output_path = os.path.join(OUTPUT_DIR, config["output_dir"])

    # 检查是否已训练过
    if os.path.exists(output_path) and os.listdir(output_path):
        has_model = any(
            f.endswith((".bin", ".safetensors"))
            for f in os.listdir(output_path)
        )
        if has_model:
            logger.info(f"[跳过] {model_key} 已有训练好的模型: {output_path}")
            return {"status": "skipped", "reason": "already_trained"}

    logger.info(f"{'='*60}")
    logger.info(f"开始训练: {config['description']}")
    logger.info(f"基础模型: {config['base_model']}")
    logger.info(f"训练样本: {len(data)} 条")
    logger.info(f"输出路径: {output_path}")
    logger.info(f"{'='*60}")

    # ---- 加载 tokenizer 和模型 ----
    logger.info("[1/5] 加载预训练模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["base_model"])
    logger.info(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ---- 创建 Dataset ----
    logger.info("[2/5] 构建训练数据集...")
    dataset = Dataset.from_list(data)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    logger.info(f"  训练集: {len(split['train'])} 条, 验证集: {len(split['test'])} 条")

    # ---- Tokenize ----
    logger.info("[3/5] Tokenize 数据集...")
    if use_mbart_preprocess:
        pre_fn = lambda x: tokenize_mbart(
            x, tokenizer, config["max_input_length"], config["max_target_length"]
        )
    else:
        pre_fn = lambda x: tokenize_standard(
            x, tokenizer, config["max_input_length"], config["max_target_length"]
        )

    tokenized_train = split["train"].map(
        pre_fn, batched=True, remove_columns=["source", "target"]
    )
    tokenized_valid = split["test"].map(
        pre_fn, batched=True, remove_columns=["source", "target"]
    )

    # ---- 训练参数 ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        eval_strategy="steps" if len(tokenized_train) >= 500 else "epoch",
        eval_steps=500,
        save_strategy="steps" if len(tokenized_train) >= 500 else "epoch",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=config["max_target_length"],
        report_to="none",
        dataloader_num_workers=0,
        # 禁用 wandb，避免网络问题
        run_name=None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ---- 开始训练 ----
    logger.info("[4/5] 开始训练...")
    start_time = time.time()
    train_result = trainer.train()
    train_time = time.time() - start_time

    # ---- 保存模型 ----
    logger.info("[5/5] 保存模型...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # 收集训练指标
    metrics = {
        "status": "completed",
        "model_key": model_key,
        "description": config["description"],
        "base_model": config["base_model"],
        "train_samples": len(split["train"]),
        "eval_samples": len(split["test"]),
        "train_time_seconds": round(train_time, 1),
        "train_loss": round(train_result.training_loss, 4) if train_result.training_loss else None,
        "eval_loss": round(train_result.metrics.get("eval_loss", 0), 4) if "eval_loss" in train_result.metrics else None,
        "epochs": config["epochs"],
        "save_path": output_path,
    }

    logger.info(f"  训练耗时: {train_time/60:.1f} 分钟")
    logger.info(f"  训练损失: {metrics['train_loss']}")
    logger.info(f"  验证损失: {metrics['eval_loss']}")
    logger.info(f"[完成] {config['description']} 训练完毕\n")

    # 清理 GPU 内存
    del model, tokenizer, trainer
    torch.cuda.empty_cache()

    return metrics


# ============================================================================
# 构建模型5训练数据 (合成: 中文原文 → 英文摘要)
# ============================================================================

def build_direct_model_data(max_samples: int = None) -> List[Dict]:
    """为模型5构造跨语言摘要训练数据

    方法: 利用中文摘要数据的原文作为输入，通过翻译模型将中文摘要翻译成英文
    作为目标输出，形成 (中文原文 → 英文摘要) 的训练对。

    这一步模拟了一个"两步走"的教师模型: 先做中文摘要，再做中→英翻译。
    """
    logger.info("=" * 60)
    logger.info("构造模型5训练数据: 中文原文 → 英文摘要")
    logger.info("=" * 60)

    # 加载中文摘要数据作为源文本
    chinese_data = load_lcsts_data(max_samples=8000)
    if len(chinese_data) < 100:
        chinese_data = load_clts_data(max_samples=8000)

    if len(chinese_data) < 50:
        logger.warning("中文摘要数据不足以构造模型5训练数据")
        return []

    # 使用预训练翻译模型 (不依赖本地微调版本)
    TRANS_MODEL = "Helsinki-NLP/opus-mt-zh-en"
    logger.info(f"  翻译教师模型: {TRANS_MODEL}")

    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline(
        "translation",
        model=TRANS_MODEL,
        device=device,
    )

    synthetic_data = []
    batch_size = 16  # 小batch避免OOM
    total = min(len(chinese_data), max_samples or len(chinese_data))

    for i in range(0, total, batch_size):
        batch = chinese_data[i : i + batch_size]
        sources = [item["source"] for item in batch]
        chinese_summaries = [item["target"] for item in batch]

        try:
            # 将中文摘要翻译为英文摘要，作为训练目标
            translations = translator(
                chinese_summaries,
                max_length=200,
                truncation=True,
            )
            en_targets = [t["translation_text"] for t in translations]

            for src, tgt in zip(sources, en_targets):
                if len(src) > 10 and len(tgt) > 3:
                    synthetic_data.append({"source": src, "target": tgt})

        except Exception as e:
            logger.warning(f"  批次 {i} 处理失败: {e}")
            continue

        if (i + batch_size) % 500 == 0:
            logger.info(f"  构造进度: {min(i + batch_size, total)}/{total}")

    del translator
    torch.cuda.empty_cache()

    logger.info(f"  合成数据: {len(synthetic_data)} 条")
    return synthetic_data


# ============================================================================
# 保存训练结果汇总
# ============================================================================

def save_training_results(all_metrics: List[dict]):
    """保存所有模型的训练指标到 JSON 文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {
        "training_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": OUTPUT_DIR,
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "models": all_metrics,
    }
    with open(RESULTS_LOG, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练结果汇总已保存: {RESULTS_LOG}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="翻译摘要系统 - 模型训练 (三个深度学习模型)"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "summarizer", "translator", "direct_model"],
        help="指定要训练的模型 (默认: all)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=MAX_SAMPLES,
        help=f"每个模型最大训练样本数 (默认: {MAX_SAMPLES})",
    )
    parser.add_argument(
        "--skip_direct", action="store_true",
        help="跳过模型5训练",
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  翻译摘要系统 - 模型训练")
    logger.info(f"  设备: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  数据目录: {DATA_DIR}")
    logger.info(f"  输出目录: {OUTPUT_DIR}")
    logger.info(f"  最大样本数: {args.max_samples}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  模型架构方案:")
    logger.info("    模型1/4 (摘要):      google/mt5-small (T5架构, Span Corruption)")
    logger.info("    模型2/3 (翻译):      Helsinki-NLP/opus-mt-zh-en (MarianMT架构)")
    logger.info("    模型5   (直接):      facebook/mbart-large-50-mmt (BART架构)")
    logger.info("    模型1=模型4, 模型2=模型3, 模型5与前四个全部不同 ✓")
    logger.info("")

    all_metrics = []

    # ==================================================================
    # 模型1/4: 多语言摘要 (mT5-small)
    # 合并中文和英文摘要数据训练同一个模型
    # ==================================================================
    if args.model in ("all", "summarizer"):
        logger.info("\n" + "#" * 60)
        logger.info("# 训练 模型1/4: 多语言摘要模型 (mT5-small)")
        logger.info("# 合并中文摘要 + 英文摘要数据")
        logger.info("#" * 60)

        lcsts = load_lcsts_data(max_samples=args.max_samples // 2)
        clts = load_clts_data(max_samples=args.max_samples // 4)
        cnn = load_cnn_dailymail_data(max_samples=args.max_samples // 4)
        news = load_news_summarization_data(max_samples=args.max_samples // 4)

        # 中文摘要数据
        cn_data = lcsts + clts
        # 英文摘要数据
        en_data = cnn + news

        logger.info(f"  中文摘要数据: {len(cn_data)} 条")
        logger.info(f"  英文摘要数据: {len(en_data)} 条")

        # 合并中英文数据训练同一个模型
        all_summ_data = cn_data + en_data
        logger.info(f"  合并总计: {len(all_summ_data)} 条")

        if len(all_summ_data) > 0:
            metrics = train_model("summarizer", all_summ_data)
            all_metrics.append(metrics)
        else:
            logger.error("摘要训练数据为空，请检查数据路径!")

    # ==================================================================
    # 模型2/3: 中英翻译 (MarianMT)
    # ==================================================================
    if args.model in ("all", "translator"):
        logger.info("\n" + "#" * 60)
        logger.info("# 训练 模型2/3: 中英翻译模型 (MarianMT)")
        logger.info("#" * 60)

        trans_data = load_wmt_translation_data(max_samples=args.max_samples)
        logger.info(f"  翻译数据总计: {len(trans_data)} 条")

        if len(trans_data) > 0:
            metrics = train_model("translator", trans_data)
            all_metrics.append(metrics)
        else:
            logger.error("翻译训练数据为空，请检查数据路径!")

    # ==================================================================
    # 模型5: 直接翻译+摘要 (mBART-mmt)
    # ==================================================================
    if args.model in ("all", "direct_model") and not args.skip_direct:
        logger.info("\n" + "#" * 60)
        logger.info("# 训练 模型5: 直接翻译摘要模型 (mBART-mmt)")
        logger.info("# 构造合成数据: 中文原文 + 翻译模型 → 英文摘要")
        logger.info("#" * 60)

        direct_data = build_direct_model_data(max_samples=args.max_samples)

        if len(direct_data) >= 100:
            metrics = train_model(
                "direct_model", direct_data, use_mbart_preprocess=True
            )
            all_metrics.append(metrics)
        else:
            logger.warning(
                "模型5训练数据不足 (需要≥100条)，跳过训练。"
                "可先确保翻译模型可用后再试。"
            )
            all_metrics.append({
                "status": "skipped",
                "reason": "insufficient_data",
                "data_count": len(direct_data),
            })

    # ==================================================================
    # 保存训练结果汇总
    # ==================================================================
    save_training_results(all_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("  全部训练流程完成!")
    logger.info(f"  模型保存在: {OUTPUT_DIR}")
    logger.info(f"  训练结果: {RESULTS_LOG}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
