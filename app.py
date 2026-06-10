"""
翻译摘要系统 - Flask Web 应用
=============================
提供 REST API 和前端界面, 支持三种翻译摘要方案。

模型方案 (满足课程设计要求):
  方案1 先摘要再翻译: 中文 → [模型1:摘要] → 中文摘要 → [模型2:翻译] → 英文摘要
  方案2 先翻译再摘要: 中文 → [模型3:翻译] → 英文 → [模型4:摘要] → 英文摘要
  方案3 直接翻译摘要: 中文 → [模型5:直接] → 英文摘要

模型约束: 模型1=模型4, 模型2=模型3, 模型5与所有其他模型不同 (共3个独立模型)
"""

import os
import sys
import io
import random
import logging
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
)

# 导入模型模块
from models import execute_approach

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask 应用初始化
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 最大 16MB 上传
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")

# 创建上传目录
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# 全局存储当前加载的文本
current_text: str = ""


# ---------------------------------------------------------------------------
# 文档读取工具
# ---------------------------------------------------------------------------
def read_txt(filepath: str) -> str:
    """读取 TXT 文件, 尝试多种编码"""
    encodings = ["utf-8", "gbk", "gb2312", "gb18030", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法读取文件 {filepath}, 尝试了编码: {encodings}")


def read_docx(filepath: str) -> str:
    """读取 Word 文档"""
    from docx import Document

    doc = Document(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def extract_text_segment(text: str, max_chars: int = 2000) -> str:
    """
    从文本中随机截取一段连续内容 (约 max_chars 字符)。
    跳过目录/标题行, 从随机位置开始, 保证内容完整性。
    """
    # 按段落分割
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    # 跳过目录/标题行 (通常很短)
    content_paragraphs = [p for p in paragraphs if len(p) > 20]

    if not content_paragraphs:
        return text[:max_chars]

    # 随机选择一个起始段落 (保证后面还有足够内容)
    start_idx = random.randint(0, len(content_paragraphs) - 1)

    # 从随机位置开始取, 凑到约 max_chars
    result = ""
    for p in content_paragraphs[start_idx:]:
        if len(result) + len(p) > max_chars:
            remaining = max_chars - len(result)
            if remaining > 100:
                cut_point = p[:remaining].rfind("。")
                if cut_point > 0:
                    result += p[: cut_point + 1]
                else:
                    result += p[:remaining]
            break
        result += p + "\n"

    # 如果随机起始点太靠后, 内容不够, 则从开头补
    if len(result) < 100 and start_idx > 0:
        return extract_text_segment(text, max_chars)

    return result.strip()


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """主页面"""
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    上传文件接口
    支持 .txt 和 .docx 格式
    """
    global current_text

    if "file" not in request.files:
        return jsonify({"error": "未选择文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    # 保存文件
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        # 根据扩展名读取
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".txt":
            raw_text = read_txt(filepath)
        elif ext in (".docx", ".doc"):
            raw_text = read_docx(filepath)
        else:
            return jsonify({"error": f"不支持的文件格式: {ext}"}), 400

        # 提取约 2000 字符的测试段落
        current_text = extract_text_segment(raw_text, max_chars=2000)

        logger.info(f"文件加载成功: {file.filename}, 提取 {len(current_text)} 字符")

        return jsonify(
            {
                "filename": file.filename,
                "total_chars": len(raw_text),
                "extracted_chars": len(current_text),
                "preview": current_text[:500] + ("..." if len(current_text) > 500 else ""),
            }
        )

    except Exception as e:
        logger.error(f"文件处理失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/load_hongloumeng", methods=["POST"])
def load_hongloumeng():
    """
    直接加载红楼梦测试段落 (2000字)
    """
    global current_text

    hlm_path = os.path.join(os.path.dirname(__file__), "红楼梦.txt")
    if not os.path.exists(hlm_path):
        return jsonify({"error": "红楼梦.txt 文件不存在"}), 404

    try:
        raw_text = read_txt(hlm_path)
        current_text = extract_text_segment(raw_text, max_chars=2000)

        logger.info(f"红楼梦加载成功, 提取 {len(current_text)} 字符")

        return jsonify(
            {
                "filename": "红楼梦.txt",
                "total_chars": len(raw_text),
                "extracted_chars": len(current_text),
                "preview": current_text[:500] + ("..." if len(current_text) > 500 else ""),
            }
        )

    except Exception as e:
        logger.error(f"加载红楼梦失败: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/execute/<int:approach_id>", methods=["POST"])
def execute(approach_id: int):
    """
    执行指定方案 (1, 2, 或 3)
    """
    global current_text

    if not current_text:
        return jsonify({"error": "请先加载文档"}), 400

    if approach_id not in (1, 2, 3):
        return jsonify({"error": f"无效的方案编号: {approach_id}"}), 400

    try:
        logger.info(f"执行方案 {approach_id}...")
        result = execute_approach(approach_id, current_text)

        return jsonify(
            {
                "approach": result["approach"],
                "original_length": result["original_length"],
                "intermediate": result["intermediate"],
                "final_result": result["final_result"],
                "final_length": result["final_length"],
            }
        )

    except Exception as e:
        logger.error(f"方案 {approach_id} 执行失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/execute_all", methods=["POST"])
def execute_all():
    """
    依次执行三种方案 (用于对比)
    """
    global current_text

    if not current_text:
        return jsonify({"error": "请先加载文档"}), 400

    results = {}
    for approach_id in [1, 2, 3]:
        try:
            logger.info(f"执行方案 {approach_id}...")
            result = execute_approach(approach_id, current_text)
            results[f"approach_{approach_id}"] = {
                "approach": result["approach"],
                "original_length": result["original_length"],
                "intermediate": result["intermediate"],
                "final_result": result["final_result"],
                "final_length": result["final_length"],
            }
        except Exception as e:
            logger.error(f"方案 {approach_id} 执行失败: {e}")
            results[f"approach_{approach_id}"] = {
                "approach": f"方案{approach_id}",
                "error": str(e),
            }

    return jsonify(results)


@app.route("/api/current_text", methods=["GET"])
def get_current_text():
    """获取当前加载的文本"""
    global current_text
    return jsonify(
        {
            "text": current_text,
            "length": len(current_text),
            "has_text": bool(current_text),
        }
    )


@app.route("/api/exit")
def exit_app():
    """退出系统"""
    import signal
    import sys

    # 返回提示页面
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head><meta charset="UTF-8"><title>系统已停止</title>
    <style>
        body { font-family: "Microsoft YaHei", sans-serif;
               display: flex; justify-content: center; align-items: center;
               height: 100vh; background: #1a1a2e; color: #e0e0e0; }
        .box { text-align: center; }
        h1 { color: #64b5f6; }
        p { color: #888; }
    </style></head>
    <body><div class="box">
        <h1>🚪 系统已停止</h1>
        <p>翻译摘要系统已退出，请关闭此页面。</p>
        <p>如需重新启动，请运行 python app.py</p>
    </div></body></html>
    """


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  翻译摘要系统 - Translation & Summarization System")
    print("  自然语言处理实践 课程设计")
    print("=" * 60)
    print()
    print("  访问地址: http://127.0.0.1:5000")
    print("  按 Ctrl+C 停止服务")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
