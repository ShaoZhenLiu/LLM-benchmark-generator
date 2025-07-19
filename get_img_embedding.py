from io import BytesIO
import base64
from PIL import Image
import json
from http import HTTPStatus
from data import get_mscoco_dataset, get_args
from tqdm import tqdm
from transformers import AutoModel


def get_embedding(examples, model):
    embedding = model.get_fused_embeddings(texts=examples["description"], images=examples["image_path"]).tolist()
    return {"embedding": embedding}


def filter_by_aspect_ratio(example):
    """
    过滤函数，保留长宽比在1.3:1到1:1.3之间的图片

    参数:
        example: datasets库中的一条数据记录，包含'image'字段(base64编码的图片)

    返回:
        bool: 如果图片长宽比在指定范围内返回True，否则返回False
    """
    try:
        # 解码base64图片
        image_data = base64.b64decode(example['image'])
        img = Image.open(BytesIO(image_data))

        # 获取图片宽高
        width, height = img.size

        # 计算长宽比(总是用较大的数除以较小的数)
        aspect_ratio = max(width, height) / min(width, height)

        # 检查长宽比是否在1.3:1范围内
        return aspect_ratio <= 1.3
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return False


if __name__ == '__main__':
    args = get_args()
    dataset = get_mscoco_dataset(args.data_path, cache_dir=args.cache_path, sample=args.sample_num)
    dataset = dataset.filter(filter_by_aspect_ratio)
    print(dataset)

    gme = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype="float16", device_map='auto', trust_remote_code=True
    )

    dataset = dataset.map(
        get_embedding,
        fn_kwargs={
            "model": gme,
        },
        batched=True,
        batch_size=args.batch_size,
        desc="get_embedding",
    )
    print(dataset)
    dataset.to_parquet(args.output_path)
