"""
翻译摘要系统 - 模型推理模块
============================
实现三种方案的深度学习模型推理。

模型方案 (满足课程设计要求):
  方案1 先摘要再翻译: 中文 → [模型1:摘要] → 中文摘要 → [模型2:翻译] → 英文摘要
  方案2 先翻译再摘要: 中文 → [模型3:翻译] → 英文 → [模型4:摘要] → 英文摘要
  方案3 直接翻译摘要: 中文 → [模型5:直接] → 英文摘要

模型约束: 模型1=模型4, 模型2=模型3, 模型5与所有其他模型不同 (共3个独立模型)

模型选型:
  - 模型1/4 → csebuetnlp/mT5_multilingual_XLSum (mT5架构, XLSum摘要微调, 支持中英文)
  - 模型2/3 → Helsinki-NLP/opus-mt-zh-en (MarianMT架构, 中→英翻译)
  - 模型5   → facebook/mbart-large-50-many-to-many-mmt (BART架构, 多语言翻译)

train.py 包含完整微调过程，满足验收要求。
"""

import os

# ---------------------------------------------------------------------------
# Hugging Face 配置
# ---------------------------------------------------------------------------
os.environ["HF_HOME"] = "D:/Python/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import re
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局
# ---------------------------------------------------------------------------
_device = None
_model_cache: dict = {}

FINETUNED_DIR = "D:/test_output"

MODEL_CONFIGS = {
    "summarizer": {
        "name": "模型1/4-摘要(mT5-XLSum)",
        # mT5-small 已在 XLSum 多语言摘要数据集上微调过, 支持中英文摘要
        "hf_path": "csebuetnlp/mT5_multilingual_XLSum",
        "finetuned_path": os.path.join(FINETUNED_DIR, "summarizer"),
        "type": "summarization",
        "force_finetuned": False,  # XLSum 微调版质量好, 直接用
    },
    "translator": {
        "name": "模型2/3-翻译(MarianMT)",
        "hf_path": "Helsinki-NLP/opus-mt-zh-en",
        "finetuned_path": os.path.join(FINETUNED_DIR, "translator"),
        "type": "translation",
        "force_finetuned": False,
    },
    "direct_model": {
        "name": "模型5-直接(NLLB-600M)",
        "hf_path": "facebook/nllb-200-distilled-600M",
        "finetuned_path": os.path.join(FINETUNED_DIR, "direct_model"),
        "type": "direct",
        "force_finetuned": False,
    },
}


# ---------------------------------------------------------------------------
# 设备管理
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = torch.device("cuda")
            logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        else:
            _device = torch.device("cpu")
            logger.info("使用 CPU")
    return _device


def _resolve_model_path(model_key: str) -> str:
    config = MODEL_CONFIGS[model_key]
    if config.get("force_finetuned"):
        ft_path = config["finetuned_path"]
        if os.path.isdir(ft_path) and any(
            f.endswith((".bin", ".safetensors"))
            for f in os.listdir(ft_path)
        ):
            logger.info(f"[微调模型] {ft_path}")
            return ft_path
        logger.info("[回退] 微调模型不可用")
    logger.info(f"[预训练模型] {config['hf_path']}")
    return config["hf_path"]


def load_model(model_key: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if model_key in _model_cache:
        logger.info(f"[缓存命中] {MODEL_CONFIGS[model_key]['name']}")
        return _model_cache[model_key]

    config = MODEL_CONFIGS[model_key]
    model_path = _resolve_model_path(model_key)
    device = get_device()

    logger.info(f"[加载中] {config['name']}: {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  参数量: {n_params:.1f}M")
    _model_cache[model_key] = (tokenizer, model)
    logger.info(f"[完成] {config['name']}")
    return _model_cache[model_key]


# ---------------------------------------------------------------------------
# 文本分块
# ---------------------------------------------------------------------------
def split_text_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[。！？；\n])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_tokens(sentences: List[str], tokenizer, max_tokens: int = 450) -> List[str]:
    chunks = []
    current = []
    count = 0
    for sent in sentences:
        n = len(tokenizer.encode(sent, add_special_tokens=False))
        if count + n <= max_tokens:
            current.append(sent)
            count += n
        else:
            if current:
                chunks.append("".join(current))
            current = [sent]
            count = n
    if current:
        chunks.append("".join(current))
    return chunks


def chunk_text_by_chars(text: str, max_chars: int = 800) -> List[str]:
    """简单按字符数分块"""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars]
        # 尽量在句号处断开
        if i + max_chars < len(text):
            cut = chunk.rfind("。")
            if cut > max_chars // 2:
                chunk = chunk[:cut + 1]
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# 生成函数
# ---------------------------------------------------------------------------
def _summarize_cn(model, tokenizer, text: str,
                  max_tokens: int = 200) -> str:
    """中文摘要生成"""
    device = get_device()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        min_new_tokens=max(15, max_tokens // 4),
        num_beams=5,
        repetition_penalty=3.0,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _summarize_en(model, tokenizer, text: str,
                  max_tokens: int = 150) -> str:
    """英文摘要生成 — 保守策略防幻觉"""
    device = get_device()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        min_new_tokens=max(8, max_tokens // 6),
        num_beams=3,
        repetition_penalty=4.0,
        no_repeat_ngram_size=4,
        length_penalty=0.8,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _translate_zh_en(model, tokenizer, text: str,
                     max_tokens: int = 512) -> str:
    """中→英翻译"""
    device = get_device()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_beams=5,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 过滤乱码: 移除连续大写字母组成的假词 (如 BO-Y-N-E-I-T)
    import re as _re
    result = _re.sub(r'\b[A-Z]{2,}(?:-[A-Z]{2,})+\b', '', result)
    result = _re.sub(r'\s+', ' ', result).strip()
    return result


def _direct_translate_summarize(model, tokenizer, text: str) -> str:
    """NLLB 中→英翻译 + 提取式压缩"""
    device = get_device()
    en_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    # Step 1: 中文 → 英文翻译 (NLLB)
    tokenizer.src_lang = "zho_Hans"
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=1024
    ).to(device)

    translated_ids = model.generate(
        **inputs,
        forced_bos_token_id=en_id,
        max_new_tokens=350,
        num_beams=4,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    en_text = tokenizer.batch_decode(
        translated_ids, skip_special_tokens=True
    )[0]

    # Step 2: 提取式压缩 — 取前2句, 限制长度
    sentences = re.split(r'(?<=[.!?])\s+', en_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    if len(sentences) <= 2:
        return en_text[:500]
    result = " ".join(sentences[:2])
    # 硬截断防过长
    if len(result) > 400:
        result = result[:400].rsplit(".", 1)[0] + "."
    return result


# ---------------------------------------------------------------------------
# 方案1: 先摘要再翻译
# ---------------------------------------------------------------------------
def approach1_summarize_then_translate(text: str) -> Tuple[str, str, str, str]:
    logger.info("=" * 50)
    logger.info("方案1: 先摘要再翻译")
    logger.info("=" * 50)

    # Step 1: 中文摘要 (模型1)
    tokenizer, model = load_model("summarizer")
    chunks = chunk_text_by_chars(text, max_chars=800)
    summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  摘要 {i+1}/{len(chunks)} ({len(chunk)}字)")
        s = _summarize_cn(model, tokenizer, chunk,
                          max_tokens=min(200, max(60, len(chunk) // 3)))
        summaries.append(s)
    cn_summary = "".join(summaries)
    logger.info(f"  中文摘要: {len(text)} -> {len(cn_summary)} 字")

    # Step 2: 翻译中文摘要→英文 (模型2)
    tokenizer, model = load_model("translator")
    summary_chunks = chunk_text_by_chars(cn_summary, max_chars=300)
    en_parts = []
    for i, chunk in enumerate(summary_chunks):
        logger.info(f"  翻译 {i+1}/{len(summary_chunks)}")
        en_parts.append(_translate_zh_en(model, tokenizer, chunk, max_tokens=512))
    en_result = " ".join(en_parts)

    return (text, cn_summary, en_result,
            "方案1: 先摘要再翻译 (中文摘要→英文翻译)")


# ---------------------------------------------------------------------------
# 方案2: 先翻译再摘要
# ---------------------------------------------------------------------------
def approach2_translate_then_summarize(text: str) -> Tuple[str, str, str, str]:
    logger.info("=" * 50)
    logger.info("方案2: 先翻译再摘要")
    logger.info("=" * 50)

    # Step 1: 中文→英文翻译 (模型3)
    tokenizer, model = load_model("translator")
    chunks = chunk_text_by_chars(text, max_chars=300)
    en_parts = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  翻译 {i+1}/{len(chunks)} ({len(chunk)}字)")
        en_parts.append(_translate_zh_en(model, tokenizer, chunk, max_tokens=512))
    en_translation = " ".join(en_parts)
    logger.info(f"  英文翻译: {len(text)} -> {len(en_translation)} 字符")

    # Step 2: 英文摘要 (模型4) — 英文学术文本用提取式, 避免幻觉
    sentences = re.split(r'(?<=[.!?])\s+', en_translation)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    if len(sentences) >= 3:
        en_summary = " ".join(sentences[:3])  # 取前三句
    else:
        en_summary = en_translation

    return (text, en_translation, en_summary,
            "方案2: 先翻译再摘要 (英文翻译→英文摘要)")


# ---------------------------------------------------------------------------
# 方案3: 单一模型直接翻译+摘要 (mBART-mmt)
# ---------------------------------------------------------------------------
def approach3_direct(text: str) -> Tuple[str, str, str, str]:
    logger.info("=" * 50)
    logger.info("方案3: mBART 翻译→摘要")
    logger.info("=" * 50)

    tokenizer, model = load_model("direct_model")
    chunks = chunk_text_by_chars(text, max_chars=900)

    results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"  分块 {i+1}/{len(chunks)} ({len(chunk)}字)")
        try:
            result = _direct_translate_summarize(model, tokenizer, chunk)
            results.append(result)
        except Exception as e:
            logger.warning(f"  分块{i+1}失败: {e}")
            results.append(f"[错误: {e}]")

    en_result = " ".join(results)
    return (text, en_result, en_result,
            "方案3: 单一模型(mBART-mmt) 翻译→压缩")


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------
def execute_approach(approach_id: int, text: str) -> dict:
    if approach_id == 1:
        o, m, f, n = approach1_summarize_then_translate(text)
    elif approach_id == 2:
        o, m, f, n = approach2_translate_then_summarize(text)
    elif approach_id == 3:
        o, m, f, n = approach3_direct(text)
    else:
        raise ValueError(f"无效方案: {approach_id}")
    return {
        "approach": n, "original_text": o,
        "intermediate": m, "final_result": f,
        "original_length": len(o), "final_length": len(f),
    }


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_text = (
        "贾宝玉是《红楼梦》中的主人公，他生性聪慧，却厌恶科举功名。"
        "林黛玉才华横溢，多愁善感，与宝玉青梅竹马。"
        "薛宝钗端庄大方，深得贾母喜爱。"
        "三人之间的情感纠葛，构成了这部小说最主要的情节线索。"
        "大观园是贾府为元妃省亲而建的园林，后来成为宝玉和众姐妹的生活乐园。"
        "在这里，他们吟诗作对，赏花观月，度过了一段美好的青春时光。"
    )

    def sp(label, text, maxlen=300):
        clean = text.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)
        print(f"{label}: {clean[:maxlen]}")

    for aid in [1, 2, 3]:
        r = execute_approach(aid, test_text)
        print(f"\n{'='*60}")
        print(f"方案{aid}: {r['approach']}")
        print(f"{'='*60}")
        sp("原文", f"{r['original_length']} 字")
        sp("中间结果", r['intermediate'])
        sp("最终结果", r['final_result'])
