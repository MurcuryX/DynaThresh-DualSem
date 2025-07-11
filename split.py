#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
text_segmenter.py

提供两种文本分段方法：
  1. split_content1：基于正则和滑窗的分层切分
  2. split_content2：基于空行的简单切分

并提供从数据目录随机读取一对文本文件 (0.txt, 1.txt) 的工具。
"""

import os
import random
import re
from typing import List, Tuple


def split_content1(text: str, chunk_size: int = 512, overlap: int = 32) -> List[str]:
    """
    分层切分：先按数字序号或多横线分割，再对过长段落做滑窗切分。
    每块长度 ≤ chunk_size，重叠 overlap 字符，最短保留 30 字。
    """
    # 按 “1. ” 或 “———” 等预分割
    parts = re.split(r'(?:^\s*\d+\.\s+|^-{3,}\s*$)', text, flags=re.MULTILINE)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            chunks.append(part)
        else:
            start = 0
            while start < len(part):
                end = start + chunk_size
                piece = part[start:end].strip()
                if len(piece) >= 30:
                    chunks.append(piece)
                start += chunk_size - overlap
    return chunks


def split_content2(text: str, min_len: int = 30) -> List[str]:
    """
    简单切分：统一换行，按两个或以上空行分段，去除小于 min_len 的段落。
    """
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'\n{2,}', normalized)
    return [p.strip() for p in parts if len(p.strip()) >= min_len]


def read_random_pair(dir_path: str) -> Tuple[str, str]:
    """
    从 dir_path 下随机选一个子文件夹，读取其中的 0.txt 和 1.txt。
    返回两段纯文本。
    """
    subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories in {dir_path}")
    chosen = random.choice(subdirs)
    base = os.path.join(dir_path, chosen)
    with open(os.path.join(base, "0.txt"), encoding="utf-8") as f0, \
         open(os.path.join(base, "1.txt"), encoding="utf-8") as f1:
        return f0.read(), f1.read()


if __name__ == "__main__":
    # 示例用法：随机读一对文件，分别用两种方法分段并打印前3块
    data_dir = "数据集地址"
    text0, text1 = read_random_pair(data_dir)

    print("=== split_content1 on text0 ===")
    for i, seg in enumerate(split_content1(text0)[:3], 1):
        print(f"[{i}] ({len(seg)}) {seg[:60]}...")

    print("\n=== split_content2 on text1 ===")
    for i, seg in enumerate(split_content2(text1)[:3], 1):
        print(f"[{i}] ({len(seg)}) {seg[:60]}...")
