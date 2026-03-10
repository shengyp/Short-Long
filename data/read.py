import pickle
import os


# 读取pkl文件的函数
def read_pkl_file(file_path):
    """
    读取.pkl文件并返回数据
    Args:
        file_path (str): pkl文件路径
    Returns:
        反序列化后的Python对象
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def inspect_pkl_structure(data, indent=0):
    """
    递归检查数据结构
    Args:
        data: 要检查的数据
        indent: 缩进级别
    """
    indent_str = "  " * indent

    if isinstance(data, dict):
        print(f"{indent_str}字典类型，包含 {len(data)} 个键")
        if len(data) > 0:
            keys = list(data.keys())[:5]  # 只显示前5个键
            print(f"{indent_str}前{len(keys)}个键: {keys}")

            # 检查第一个键对应的值类型
            first_key = list(data.keys())[0]
            print(f"{indent_str}键 '{first_key}' 的值类型: {type(data[first_key])}")
            inspect_pkl_structure(data[first_key], indent + 1)
    elif isinstance(data, list):
        print(f"{indent_str}列表类型，长度: {len(data)}")
        if len(data) > 0:
            print(f"{indent_str}第一个元素的类型: {type(data[0])}")
            inspect_pkl_structure(data[0], indent + 1)
    elif isinstance(data, tuple):
        print(f"{indent_str}元组类型，长度: {len(data)}")
        if len(data) > 0:
            print(f"{indent_str}第一个元素的类型: {type(data[0])}")
            inspect_pkl_structure(data[0], indent + 1)
    elif hasattr(data, 'shape'):  # 对于numpy数组或torch张量
        print(f"{indent_str}数组类型，形状: {data.shape}")
        print(f"{indent_str}数据类型: {data.dtype}")
    else:
        print(f"{indent_str}类型: {type(data)}")
        if indent == 0:
            print(f"{indent_str}数据示例: {data}")


if __name__ == "__main__":
    # 文件路径
    file_path = "data/user_post_embeddings_bert_wwm.pkl"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        print("请确认文件路径是否正确，或者尝试以下路径：")
        print("1. 'user_post_embeddings_bert_wwm.pkl'")
        print("2. './data/weibo_bert/user_post_embeddings_bert_wwm.pkl'")
        print("3. 提供文件的完整路径")

        # 尝试其他可能路径
        possible_paths = [
            "user_post_embeddings_bert_wwm.pkl",
            "./data/weibo_bert/user_post_embeddings_bert_wwm.pkl",
            "../data/weibo_bert/user_post_embeddings_bert_wwm.pkl"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                print(f"找到文件: {file_path}")
                break
    else:
        print(f"找到文件: {file_path}")

    # 读取pkl文件
    print("\n正在读取文件...")
    data = read_pkl_file(file_path)

    if data is not None:
        print(f"\n文件读取成功！")
        print("=" * 50)

        # 1. 显示数据基本信息
        print("数据基本信息：")
        print(f"数据类型: {type(data)}")

        # 2. 检查数据结构
        print("\n数据结构分析：")
        inspect_pkl_structure(data)

        # 3. 尝试显示前3个数据
        print("\n" + "=" * 50)
        print("尝试显示前3个数据：")

        if isinstance(data, dict):
            print("数据是字典格式，显示前3个键值对：")
            items = list(data.items())
            for i, (key, value) in enumerate(items[:3]):
                print(f"\n[{i + 1}] 键: {key}")
                print(f"    值类型: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"    形状: {value.shape}")
                    print(f"    数据类型: {value.dtype}")
                    if value.shape[0] > 0:
                        print(f"    前5个值: {value[:5] if len(value.shape) == 1 else value[0][:5]}")
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    print(f"    前3个元素: {value[:3]}")
                else:
                    print(f"    值: {value}")

        elif isinstance(data, list):
            print("数据是列表格式，显示前3个元素：")
            for i, item in enumerate(data[:3]):
                print(f"\n[{i + 1}] 类型: {type(item)}")
                if hasattr(item, 'shape'):
                    print(f"    形状: {item.shape}")
                    print(f"    数据类型: {item.dtype}")
                    if item.shape[0] > 0:
                        print(f"    前5个值: {item[:5] if len(item.shape) == 1 else item[0][:5]}")
                elif isinstance(item, (dict, list, tuple)):
                    print(f"    内容: {item}")
                else:
                    print(f"    值: {item}")

        elif isinstance(data, tuple):
            print("数据是元组格式，显示前3个元素：")
            for i, item in enumerate(data[:3]):
                print(f"\n[{i + 1}] 类型: {type(item)}")
                print(f"    值: {item}")

        elif hasattr(data, 'shape'):  # numpy数组或torch张量
            print(f"数据是数组格式，形状: {data.shape}")
            print(f"显示前1行数据：")
            for i in range(min(1, data.shape[0])):
                print(f"\n[{i + 1}] 第{i}行: {data[i] if len(data.shape) == 1 else data[i][:10]}...")

        else:
            print(f"数据显示: {data}")

        print("\n" + "=" * 50)
        print(f"数据总览完成！")