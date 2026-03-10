import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import traceback
from models.model import PostLevel_GRU_Model
from data.data_loader import read_embedding_data, EmbeddingDataset, collate_fn_embeddings

# 有序回归损失
try:
    from utils.order_loss import loss_function as ordinal_loss_fn
    from utils.order_loss import gr_metrics
except ImportError:
    print("警告: 未找到 utils.order_loss, 只有Weibo/SuicidEmoji模式可以运行")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="统一训练脚本")

    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['weibo', 'reddit', 'bigdata', 'suicidEmoji'],
                        help='选择数据集类型')
    # 消融实验参数
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_stef', 'no_ltef', 'mlp_fusion'],
                        help='选择消融实验模式')

    parser.add_argument("--data_path", type=str, required=True, help='数据路径')
    parser.add_argument("--val_path", type=str, default=None, help='验证数据路径')
    parser.add_argument("--test_path", type=str, default=None, help='测试数据路径')
    parser.add_argument("--save_path", type=str, default='./models/best_model.pth', help='模型保存路径')

    # 模型与训练参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=768)
    parser.add_argument("--gru_size", type=int, default=128)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--seed", default=2024, type=int)

    args = parser.parse_args()

    # 设置类别数
    if args.dataset_type == 'reddit':
        args.class_num = 5
    elif args.dataset_type == 'bigdata':
        args.class_num = 4  # Bigdata 为 4 分类
    else:
        # weibo 和 suicidEmoji 都是二分类
        args.class_num = 2

    return args


def evaluate(model, iterator, device, dataset_type, is_test=False):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in iterator:
            inputs, labels, post_masks = batch
            inputs, labels, post_masks = inputs.to(device), labels.to(device), post_masks.to(device)

            output = model(inputs, post_masks)
            _, predicted = torch.max(output.data, 1)

            preds.extend(predicted.cpu().numpy().tolist())
            trues.extend(labels.cpu().numpy().tolist())

    # 计算评价指标
    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average='macro')
    weighted_f1 = f1_score(trues, preds, average='weighted')

    GP, GR, FS, OE = 0.0, 0.0, 0.0, 0.0
    bi_prec, bi_rec, bi_f1 = 0.0, 0.0, 0.0
    main_score = 0.0
    log_str = ""

    # 根据任务类型计算特定指标
    if dataset_type in ['weibo', 'suicidEmoji']:
        # 二分类任务
        bi_prec = precision_score(trues, preds, pos_label=1, zero_division=0)
        bi_rec = recall_score(trues, preds, pos_label=1, zero_division=0)
        bi_f1 = f1_score(trues, preds, pos_label=1, zero_division=0)

        main_score = bi_f1
        log_str = f"Eval -> Acc: {acc:.4f} | F1: {bi_f1:.4f} | W-F1: {weighted_f1:.4f}"


    elif dataset_type in ['reddit', 'bigdata']:
        try:
            GP, GR, FS, OE = gr_metrics(preds, trues)
        except Exception as e:
            print(f"❌ 计算指标时发生错误: {e}")
            traceback.print_exc()
            GP, GR, FS, OE = 0.0, 0.0, 0.0, 0.0

        if dataset_type == 'bigdata':
            main_score = macro_f1
            log_str = f"Eval -> Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | W-F1: {weighted_f1:.4f}"
        elif dataset_type == 'reddit':
            main_score = FS
            log_str = f"Eval -> Acc: {acc:.4f} | FScore: {FS:.4f} | W-F1: {weighted_f1:.4f}"

    metrics = {'Acc': acc, 'Macro-F1': macro_f1, 'Weighted-F1': weighted_f1, 'FS': FS}

    #输出打印
    if is_test:
        print("=" * 40)
        print(f"Test Accuracy (Acc):   {acc:.4f}")

        if dataset_type in ['weibo', 'suicidEmoji']:
            print(f"Test Precision (Pos):  {bi_prec:.4f}")
            print(f"Test Recall (Pos):     {bi_rec:.4f}")
            print(f"Test F1 (Pos):         {bi_f1:.4f}")
        else:
            print(f"Test Precision (GP):   {GP:.4f}")
            print(f"Test Recall (GR):      {GR:.4f}")
            print(f"Test FScore (FS):      {FS:.4f}")
            print(f"Test Error (OE):       {OE:.4f}")

        print("-" * 20)
        print(f"Test Macro-F1:         {macro_f1:.4f}")
        print(f"Test Weighted-F1:      {weighted_f1:.4f}")
        print("=" * 40)
    else:
        print(log_str)

    return metrics, main_score


def train(args, model, train_loader, val_loader, optimizer, device):
    # 初始化为 -1，确保F1 为 0 也能保存一次模型
    best_score = -1.0

    # 损失函数选择
    if args.dataset_type in ['weibo', 'suicidEmoji']:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            inputs, labels, post_masks = batch
            inputs, labels, post_masks = inputs.to(device), labels.to(device), post_masks.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, post_masks)

            if args.dataset_type in ['weibo', 'suicidEmoji']:
                loss = criterion(outputs, labels)
            else:
                loss = ordinal_loss_fn(outputs, labels, loss_type='ordinal', expt_type=args.class_num)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

        metrics, current_score = evaluate(model, val_loader, device, args.dataset_type)

        if current_score >= best_score:
            best_score = current_score
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"保存新最佳模型! (Score: {best_score:.4f}) -> {args.save_path}")


def main():
    args = parse_args()

    # 更新保存路径文件名
    dir_name, file_name = os.path.split(args.save_path)
    base_name, ext = os.path.splitext(file_name)
    new_file_name = f"{base_name}_{args.dataset_type}_{args.ablation}{ext}"
    args.save_path = os.path.join(dir_name, new_file_name)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device} | 模式: {args.dataset_type} | 消融: {args.ablation} | 类别数: {args.class_num}')

    # 数据读取逻辑
    if args.dataset_type in ['bigdata', 'suicidEmoji']:
        if not (args.val_path and args.test_path):
            raise ValueError(f"{args.dataset_type} 模式必须提供 --val_path 和 --test_path")

        print("正在读取独立的 Train/Val/Test 文件...")
        train_data = read_embedding_data(args.data_path)
        val_data = read_embedding_data(args.val_path)
        test_data = read_embedding_data(args.test_path)
    else:
        # 1. 读取数据
        all_data = read_embedding_data(args.data_path)

        print("正在根据 Embedding 内容进行确定性排序...")

        def get_sort_key(item):
            # 获取该样本的 embeddings 列表
            embs = item.get('embeddings')
            # 如果列表非空，且第一个向量非空，取第一个向量的第一个数值
            if embs is not None and len(embs) > 0 and len(embs[0]) > 0:
                return float(embs[0][0])
            # 如果是空数据，返回 0 (排在最前面)
            return 0.0

        # 执行排序：改变 all_data 的物理顺序，使其固定下来
        all_data.sort(key=get_sort_key)

        labels = [item['label'] for item in all_data]

        train_data, temp_data, train_labels, temp_labels = train_test_split(
            all_data, labels, stratify=labels, test_size=0.2, random_state=args.seed
        )
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, stratify=temp_labels, test_size=0.5, random_state=args.seed
        )

    print(f"数据量 -> Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = torch.utils.data.DataLoader(EmbeddingDataset(train_data), batch_size=args.batch_size, shuffle=True,
                                               collate_fn=collate_fn_embeddings)
    val_loader = torch.utils.data.DataLoader(EmbeddingDataset(val_data), batch_size=args.batch_size, shuffle=False,
                                             collate_fn=collate_fn_embeddings)
    test_loader = torch.utils.data.DataLoader(EmbeddingDataset(test_data), batch_size=args.batch_size, shuffle=False,
                                              collate_fn=collate_fn_embeddings)

    # 初始化模型
    model = PostLevel_GRU_Model(args, device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 训练
    print("开始训练...")
    train(args, model, train_loader, val_loader, optimizer, device)

    # 测试
    print("\n加载最佳模型进行最终测试...")

    # 文件存在性检查
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path))
        evaluate(model, test_loader, device, args.dataset_type, is_test=True)
    else:
        print(f"⚠️ 警告: 未找到保存的模型文件 {args.save_path}。可能是因为训练过程中模型未能收敛或未能保存。")
        print("尝试使用当前（最后一轮）的模型参数进行测试...")
        evaluate(model, test_loader, device, args.dataset_type, is_test=True)


if __name__ == '__main__':
    main()