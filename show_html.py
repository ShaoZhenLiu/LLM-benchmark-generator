import base64
import os

from datasets import load_dataset


def generate_html(dataset1, attr_name1, dataset2, attr_name2,
                  dataset3, attr_name3, dataset4, attr_name4, output_file="output.html"):
    """
    生成包含图片和四组文本的HTML文件

    参数:
    dataset1-4: 包含字典的列表，每个字典包含'image_path'和文本属性
    attr_name1-4: 每个数据集中文本属性的名称
    output_file: 输出HTML文件名
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image and Text Display</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .image-container { text-align: center; margin: 20px 0; }
            .image-container img { max-width: 80%; border: 1px solid #ddd; border-radius: 4px; }
            .text-grid { 
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 30px;
            }
            .text-item { 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 4px; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .dataset-title {
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
                font-size: 18px;
            }
            .text-content {
                font-size: 16px;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <h1>Image and Text Display</h1>
    """

    for i in range(len(dataset1)):
        # 获取图片路径（所有数据集相同，取第一个即可）
        img_path = dataset1[i].get('image_path', '')

        # 将图片转换为Base64编码
        try:
            with open(img_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                img_src = f"data:image/{os.path.splitext(img_path)[1][1:]};base64,{encoded_image}"
        except:
            img_src = ""  # 如果图片不存在，使用空字符串

        # 图片部分
        html_content += f"""
        <div class="image-container">
            <h2>Image {i + 1}</h2>
            <img src="{img_src}" alt="Image {i + 1}">
        </div>
        """

        # 文本部分网格
        html_content += "<div class='text-grid'>"

        # 处理四个数据集的文本
        datasets = [dataset1, dataset2, dataset3, dataset4]
        attr_names = [attr_name1, attr_name2, attr_name3, attr_name4]

        for idx, (data, attr) in enumerate(zip(datasets, attr_names)):
            # 获取文本内容
            text_data = data[i].get(attr, '')

            # 处理列表类型文本
            if isinstance(text_data, list):
                text_content = "<br>".join(text_data)
            else:
                text_content = str(text_data)

            html_content += f"""
            <div class="text-item">
                <div class="dataset-title">Dataset {list(data.download_checksums.keys())[0].split("/")[-1]}</div>
                <div class="text-content">{text_content}</div>
            </div>
            """

        html_content += "</div>"  # 关闭text-grid

    html_content += """
    </body>
    </html>
    """

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML文件已生成: {output_file}")


# 示例用法
if __name__ == "__main__":
    data1 = load_dataset("json", data_files="./data/t2i_long_text.jsonl", split="train")  # caption
    data2 = load_dataset("json", data_files="./data/t2i_complex_semantic.jsonl", split="train")  # instructions
    data3 = load_dataset("json", data_files="./data/edit_long_text.jsonl", split="train")  # edit_instructions
    data4 = load_dataset("json", data_files="./data/edit_complex_semantic.jsonl", split="train")  # seq_instructions

    # 调用函数生成HTML
    generate_html(
        dataset1 = data1, attr_name1 = 'instructions',
        dataset2 = data2, attr_name2 = 'instructions',
        dataset3 = data3, attr_name3 = 'instructions',
        dataset4 = data4, attr_name4 = 'instructions',
        output_file = "image_text_display.html"
    )