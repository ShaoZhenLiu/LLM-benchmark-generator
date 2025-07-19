import os
import re
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import requests
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm

from prompt import t2i_long_text_prompt, t2i_complex_prompt, edit_long_text_prompt, edit_complex_prompt
from data import image_to_base64, parse_resp, get_args


PROMPT_TYPE = {
    "t2i_long_text": t2i_long_text_prompt,
    "t2i_complex_semantic": t2i_complex_prompt,
    "edit_long_text": edit_long_text_prompt,
    "edit_complex_semantic": edit_complex_prompt,
}

key = "EMPTY"
url = "http://0.0.0.0:18901/v1"
client = OpenAI(
    api_key=key,
    base_url=url,
)
try:
    response = requests.get(f"{url}/models")
    eval_model_name = response.json()['data'][0]['id']
except Exception as e:
    print(e)


def process(args):
    image_data, description, task_type = args
    completion = client.chat.completions.create(
        model=eval_model_name,  # 选择目标模型
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": image_data}},
            {"type": "text", "text": PROMPT_TYPE[task_type].format(description=description)},
        ]}],
        max_tokens=1024
    )
    return completion


def get_resp(examples, task_type, num_workers=8):
    image_data_ls = [image_to_base64(image_path) for image_path in examples["image_path"]]
    description_ls = examples["description"]
    process_args = [(img, des, task_type) for img, des in zip(image_data_ls, description_ls)]
    
    pool = multiprocessing.Pool(processes=num_workers)
    completion_ls = []
    with tqdm(total=len(process_args), desc=f"{task_type} generation") as pbar:
        for result in pool.imap(process, process_args):
            if result is not None:
                completion_ls.append(result)
                pbar.update(1)

    pool.close()
    pool.join()
    
    # print(completion.choices[0].message.content)
    # print(completion.usage.completion_tokens)
    response_ls = [completion.choices[0].message.content for completion in completion_ls]
    token_length_ls = [completion.usage.completion_tokens for completion in completion_ls]
    # print(response_ls)
    # print(token_length_ls)
    extract_type = "answer_only" if task_type in ["t2i_complex_semantic", "edit_complex_semantic"] else "classic"
    task_type_ls, instruction_ls = map(list, zip(*[parse_resp(response) for response in response_ls]))  # 看看这里对不对
    return {
        "task_type": task_type_ls,
        "instructions": instruction_ls,
        "completion_tokens": token_length_ls,
        "full_response": response_ls,
    }
    # print(completion.to_json())


if __name__ == '__main__':
    args = get_args()
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    task_type = ["t2i_long_text", "t2i_complex_semantic", "edit_long_text", "edit_complex_semantic"]
    for task in task_type:
        caption_dataset = dataset.map(
            get_resp,
            fn_kwargs={
                "num_workers": args.batch_size,
                "task_type": task,
            },
            batched=True,
            batch_size=len(dataset)
        )
        print(f"{task} generation done!")
        cur_output_path = os.path.join(args.output_path, f"{task}.jsonl")

        caption_dataset.to_json(cur_output_path)

