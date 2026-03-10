import os
# 设置 Hugging Face 镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# 原始 txt 文件路径
RAW_FILES = {
    "train": "SuicidEmoji/train.txt",
    "val": "SuicidEmoji/val.txt",
    "test": "SuicidEmoji/test.txt"
}

# 输出 pkl 文件路径
OUTPUT_FILES = {
    "train": "SuicidEmoji/suicidEmoji_train_bert.pkl",
    "val": "SuicidEmoji/suicidEmoji_val_bert.pkl",
    "test": "SuicidEmoji/suicidEmoji_test_bert.pkl"
}

BERT_MODEL = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128


# =========================================

class BertExtractor:
    def __init__(self):
        print(f"Loading BERT ({DEVICE})...")
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.model = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, text):
        inputs = self.tokenizer(
            str(text),
            padding='max_length',
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = self.model(**inputs)
        # 获取 [CLS] 向量 (1, 768)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def parse_line(line):
    """
    解析每一行: 文本 + 标签 找到最后一个空格/Tab，右边是 Label，左边是 Text
    """
    line = line.strip()
    if not line: return None, None

    # 从右边切分一次，最后一位是标签
    parts = line.rsplit(maxsplit=1)
    if len(parts) != 2:
        return None, None

    text, label_str = parts
    try:
        label = int(label_str)
    except ValueError:
        return None, None  # 标签解析失败跳过

    return text, label


def process_file(extractor, input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ 跳过 (文件不存在): {input_path}")
        return

    print(f"正在处理: {input_path} ...")
    data_list = []

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        text, label = parse_line(line)
        if text is None: continue

        # 转换为向量
        emb = extractor.get_embedding(text)  # shape (1, 768)

        # 封装成模型需要的格式
        data_list.append({
            'user': f"user_{idx}",  # 虚拟 ID
            'label': label,
            'embeddings': emb  # (1, 768) 的 numpy 数组
        })

    print(f"保存到 {output_path} (共 {len(data_list)} 条)...")
    with open(output_path, 'wb') as f:
        pickle.dump(data_list, f)


if __name__ == "__main__":
    extractor = BertExtractor()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(list(OUTPUT_FILES.values())[0]), exist_ok=True)

    for key in RAW_FILES:
        process_file(extractor, RAW_FILES[key], OUTPUT_FILES[key])

    print("\n✅ 所有数据处理完成！")





