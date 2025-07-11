from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from split import read_random_task_document, split_content1, split_content2
from split_match import MatchGenerator
import re


class ColonBasedMatcher:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def _split_by_colon(self, text: str) -> List[str]:
        """
        精准冒号段落切分算法（段首第一个冒号不分割）
        规则：
        1. 第一个冒号行不触发分割，后续冒号行触发新段落
        2. 每个冒号行的前一行作为标题行
        3. 空冒号行合并到前一段落
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return []

        # 识别所有冒号行（包含空冒号）
        colon_indices = [i for i, line in enumerate(lines) if re.search(r'[:：]', line)]
        colon_indices.insert(0, 0)

        # 没有冒号行或只有一个冒号行时返回全文
        if len(colon_indices) < 2:
            return ["\n".join(lines)]

        # 从第二个冒号开始处理（跳过第一个）
        split_points = colon_indices[1:]

        paragraphs = []

        current = 0

        # 处理后续段落
        for i in range(0, len(split_points)):
            next_point = split_points[i + 1] if i < len(split_points) - 1 else len(lines)

            if next_point == len(lines):
                paragraphs.append("\n".join(lines[current:next_point]))
                break

            # 处理空冒号行
            if lines[next_point].startswith('：') or lines[next_point].startswith(':'):
                paragraphs.append("\n".join(lines[current:next_point - 1]))
                current = next_point - 1
            else:
                paragraphs.append("\n".join(lines[current:next_point]))
                current = next_point

        if len(paragraphs) >= 3 and paragraphs[1].strip().endswith((":", "：")):
            merged = paragraphs[1] + '\n' + paragraphs[2]
            paragraphs = [paragraphs[0], merged] + paragraphs[3:]

        return paragraphs

    def enhanced_pair_match(self, content1_top_paras: List[str], content2_para: str) -> List[Tuple[str, List[dict]]]:
        """
        改进后的匹配流程（保持原有接口）
        """
        # 1. 按冒号切分content2段落
        content2_segments = self._split_by_colon(content2_para)

        # 2. 生成content1的组合（保持原有5行窗口逻辑）
        # 初始化content1_pairs和pair_origin列表，用于存储处理后的段落和原始索引信息
        content1_pairs = []
        pair_origin = []

        # 遍历content1_top_paras中的每个段落及其索引
        for para_idx, para in enumerate(content1_top_paras):
            # 将段落按行分割，去除每行两端的空白字符，并过滤掉空行
            lines = [line.strip() for line in para.split('\n') if line.strip()]

            # 生成5行窗口（步长3），即每次取5行作为一个块，每3行移动一次窗口
            # 这样做可以减少数据量，同时保持段落的上下文信息
            chunks = ['\n'.join(lines[i:i + 5]) for i in range(0, len(lines), 3) if i + 5 <= len(lines)]

            if not chunks:
                # fallback：如果一个段落都没拆出chunk，至少把它自己当一个chunk
                chunks = [para]
                origin_idxs = [para_idx]
            else:
                origin_idxs = [para_idx] * len(chunks)

            # 将生成的块添加到content1_pairs列表中
            content1_pairs.extend(chunks)

            # 记录每个块来源于哪个段落，便于后续处理
            pair_origin.extend(origin_idxs)

        # 3. 批量编码
        content1_emb = self.model.encode(content1_pairs, convert_to_tensor=True)
        content2_emb = self.model.encode(content2_segments, convert_to_tensor=True)

        # 4. 相似度计算与动态对齐
        sim_matrix = np.inner(content2_emb.cpu(), content1_emb.cpu())

        # 5. 带约束的匹配（前向窗口=3）
        results = []
        prev_match_idx = 0
        for seg_idx in range(len(content2_segments)):
            # 限制搜索范围
            start = max(0, prev_match_idx - 1)
            end = min(sim_matrix.shape[1], prev_match_idx + 4)
            best_idx = np.argmax(sim_matrix[seg_idx, start:end]) + start

            matches = []
            if best_idx < len(pair_origin):
                origin_idx = pair_origin[best_idx]
                matches.append({
                    "content1_pair": content1_pairs[best_idx],
                    "similarity": float(sim_matrix[seg_idx][best_idx]),
                    "source_para": content1_top_paras[origin_idx],
                    "source_idx": origin_idx
                })
                prev_match_idx = best_idx

            results.append((
                content2_segments[seg_idx],
                sorted(matches, key=lambda x: x['similarity'], reverse=True)[:2]
            ))

        return results


# 使用示例
if __name__ == "__main__":
    # 初始化匹配器
    matcher = ColonBasedMatcher()  # 补全括号创建实例

    # 读取并分割文档
    for i in range(10):
        content1, content2 = read_random_task_document("generate_text/doc_binary/0")
        content1_paras = split_content1(content1)
        content2_paras = split_content2(content2)

        # 生成段落级匹配矩阵
        generator = MatchGenerator()
        sim_matrix = generator.build_matrix(content1_paras, content2_paras)
        para_matches = generator.get_top_matches(sim_matrix, content1_paras, content2_paras, top_k=2)

        # 遍历每个content2段落进行两行组合匹配
        for para_match in para_matches:
            content2_para = para_match['source_para']
            content1_top_paras = [match['target_para'] for match in para_match["matches"]]

            print("\n" + "=" * 60)
            print(f"▌Content2当前段落：\n{content2_para}")
            print("=" * 60)

            # 执行两行组合匹配
            pair_matches = matcher.enhanced_pair_match(content1_top_paras, content2_para)

            if not pair_matches:
                print("⚠ 未找到有效匹配组合")
                continue

            # 打印匹配结果
            for pair_idx, (content2_pair, match_list) in enumerate(pair_matches, 1):
                print(f"\n▌组合{pair_idx}: {content2_pair.replace('[SEP]', ' ➔ ')}")

                for match_idx, match in enumerate(match_list, 1):
                    print(f"  🏅Top-{match_idx} 匹配:")
                    print(f"  相似度: {match['similarity']:.2%}")
                    # print(f"  来源段落: {match['source_para']}")
                    print(f"  【匹配内容】: {match['content1_pair'].replace('[SEP]', ' ⇨ ')}")
                    print("-" * 60)