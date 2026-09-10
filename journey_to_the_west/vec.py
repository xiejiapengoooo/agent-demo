import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 800


model = SentenceTransformer(MODEL_NAME)


def split_text(text: str, chunk_size=CHUNK_SIZE):
    """优先按换行切分，超长段落再按更细的分隔符递归切分。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    separators = ("\n\n", "\n", "。", "！", "？", "；", "，")

    def split_recursively(content, separator_index=0):
        if len(content) <= chunk_size:
            return [content]

        if separator_index >= len(separators):
            return [
                content[start : start + chunk_size]
                for start in range(0, len(content), chunk_size)
            ]

        separator = separators[separator_index]
        parts = content.split(separator)

        # 当前分隔符不存在时，继续尝试更细粒度的分隔符。
        if len(parts) == 1:
            return split_recursively(content, separator_index + 1)

        result = []
        for index, part in enumerate(parts):
            if index < len(parts) - 1:
                part += separator
            if part:
                result.extend(split_recursively(part, separator_index + 1))
        return result

    pieces = split_recursively(text)
    chunks = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue

        chunk = current.strip()
        if chunk:
            chunks.append(chunk)

        current = piece

    chunk = current.strip()
    if chunk:
        chunks.append(chunk)

    return chunks


with open("journey_to_the_west.txt", "r", encoding="gb18030") as file:
    text = file.read()


chunks = split_text(text)

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    chunks,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)

embeddings = np.asarray(embeddings, dtype="float32")

# 向量归一化后，内积搜索等价于余弦相似度搜索
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# 保存 FAISS 索引
faiss.write_index(index, str(BASE_DIR / "journey_to_the_west.index"))

# 保存文本块和建库配置
store = {
    "embedding_model": MODEL_NAME,
    "chunk_size": CHUNK_SIZE,
    "chunks": chunks,
}

with open(BASE_DIR / "chunks.json", "w", encoding="utf-8") as file:
    json.dump(store, file, ensure_ascii=False, indent=2)

print(f"建库完成，共写入 {index.ntotal} 条向量")
