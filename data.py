import os
import re
from io import BytesIO
import base64
from pathlib import Path
import argparse
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import requests
from PIL import Image
from datasets import Dataset, load_dataset
from tqdm import tqdm


def get_image(args):
    """下载图像或从缓存加载，返回PIL图像对象"""
    # 生成缓存文件名
    # filename = hashlib.md5(url.encode()).hexdigest() + ".jpg"
    url, cache_dir = args
    filename = url.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)

    # 检查缓存是否存在
    if os.path.exists(cache_path):
        try:
            return Image.open(cache_path), cache_path
        except:
            # 无效图像文件则删除并重新下载
            os.remove(cache_path)

    # 下载图片
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.save(cache_path)
        return img, cache_path
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None, None


def get_mscoco_dataset(data_file, cache_dir="./data/image_cache", sample=None):
    """
    加载并处理MSCoco数据集，返回包含图像和合并文本的数据集

    参数:
    data_file (str): Parquet数据文件路径
    cache_dir (str): 图像缓存目录(默认为"./image_cache")

    返回:
    Dataset: 处理后的数据集，包含'image'和'description'列
    """
    # 1. 加载数据集
    dataset = load_dataset("parquet", data_files=data_file)["train"]

    # 2. 合并相同URL的文本描述
    def merge_texts(examples):
        merged = {}
        for url, desc in zip(examples["URL"], examples["TEXT"]):
            if url not in merged:
                merged[url] = []
            merged[url].append(desc)
        return {
            "url": list(merged.keys()),
            "description": ["\n".join(texts) for texts in merged.values()]
        }

    merged_dataset = dataset.map(
        merge_texts,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names
    )

    if sample is not None:
        merged_dataset = merged_dataset.select(range(sample))

    # 3. 下载图片并缓存处理
    os.makedirs(cache_dir, exist_ok=True)

    def add_images(examples, num_workers=16):
        images = []
        images_paths = []

        pool = multiprocessing.Pool(processes=num_workers)
        process_args = [(url, cache_dir) for url in examples["url"]]
        with tqdm(total=len(process_args), desc="Processing text for each image") as pbar:
            for result in pool.imap(get_image, process_args):
                if result is not None:
                    raw_image = result[0]
                    if raw_image is not None:
                        buffered = BytesIO()
                        raw_image.save(buffered, format="JPEG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    images.append(img_base64 if img_base64 else None)
                    images_paths.append(result[1])
                    pbar.update(1)

        pool.close()
        pool.join()

        return {"image": images, "image_path": images_paths}

    # 添加图像列
    final_dataset = merged_dataset.map(
        add_images,
        batched=True,
        batch_size=len(merged_dataset),
    )

    # 4. 过滤掉下载失败的图像
    return final_dataset.filter(lambda x: x["image"] is not None)


def image_to_base64(image_path: str) -> str:
    """
    将图片文件编码为Base64字符串

    Args:
        image_path: 图片文件路径

    Returns:
        Base64编码的字符串（带前缀格式：'data:image/{ext};base64,...'）

    Raises:
        FileNotFoundError: 当图片文件不存在时
        ValueError: 当文件不是图片或无法读取时
    """
    # 验证文件是否存在
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    # 获取文件扩展名（不带点）
    ext = Path(image_path).suffix[1:].lower()
    if ext not in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
        raise ValueError(f"不支持的图片格式: .{ext}")

    # 读取文件并编码
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/{ext};base64,{encoded_string}"
    except Exception as e:
        raise ValueError(f"图片读取失败: {str(e)}")


task_pattern = re.compile(r'<task>(.*?)</task>', re.DOTALL)
answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)

def parse_resp(resp, extract_type="classic"):
    """
    从输入字符串中提取 <task>...</task> 和 <answer>...</answer> 中的内容

    参数:
        resp (str): 包含 XML 样式标签的输入字符串
        extract_type (str): classic or answer_only

    返回:
        tuple: (task_content, answer_content) 如果两个标签都存在
               (task_content, None) 如果只有 task 标签存在
               (None, answer_content) 如果只有 answer 标签存在
               (None, None) 如果两个标签都不存在
    """
    # 搜索匹配内容
    task_match = task_pattern.search(resp)
    answer_match = answer_pattern.search(resp)

    # 提取内容，如果没有匹配则返回 None
    task_content = task_match.group(1).strip() if task_match else None
    answer_content = answer_match.group(1).strip() if answer_match else None
    if ((task_content is None) or (answer_content is None)) and extract_type == "classic":
        print("An parsing error occurred.1")
        print(task_content)
        print(answer_content)
        print("="*30)
    elif (answer_content is None) and extract_type == "answer_only":
        print("An parsing error occurred.2")
        print(task_content)
        print(answer_content)
        print("="*30)
    return task_content, answer_content


def get_args():
    parser = argparse.ArgumentParser(description='处理数据并请求模型')

    parser.add_argument('--data_path', type=str, default='./data/mscoco.parquet',
                        help='数据文件路径')
    parser.add_argument('--cache_path', type=str, default='./data/image_cache',
                        help='数据文件路径')
    parser.add_argument('--model_path', type=str, default='Alibaba-NLP/gme-Qwen2-VL-7B-Instruct',
                        help='加载模型名字或路径')
    parser.add_argument('--sample_num', type=int, default=1e4,
                        help='要处理的数据样本个数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='每次处理的样本批量')
    parser.add_argument('--output_path', type=str, default='./data/tmp.jsonl',
                        help='结果输出路径')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # 使用示例
    dataset = get_mscoco_dataset(args.data_path, cache_dir=args.cache_path, sample=args.sample_num)
    print(dataset)

    # 测试输出
    print(f"数据集包含 {len(dataset)} 个样本")

    if len(dataset) > 0:
        sample = dataset[0]

        print("\n合并描述:")
        print(sample["description"])

        print("\nBase64图像(前100字符):")
        print(sample["image"][:100] + "...")