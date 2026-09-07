import torch
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class CaseDateset(Dataset):
    def __init__(self, features: list[str], tokenizer: str, max_length):
        """
        Args:
            - features: PlainText list
            - tokenizer: The path of tokenizer
            - max_length: Control the maximum number of tokens
        """
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.all_token_ids, self.all_attention_masks, self.all_label_ids = self.init_dataset()

    def __getitem__(self, idx):
        return self.all_token_ids[idx], self.all_label_ids[idx], self.all_attention_masks[idx]
        # return self.all_token_ids[idx], self.all_label_ids[idx], self.all_attention_masks[idx], self.all_vaild_lens[idx]

    def __len__(self):
        return len(self.all_token_ids)

    def init_dataset(self):
        all_token_ids, all_attention_masks, all_label_ids = self._preprocess(self.features)

        return all_token_ids, all_attention_masks, all_label_ids

    def _preprocess(self, features):
        all_token_ids = []
        all_attention_masks = []
        all_label_ids = []

        with mp.Pool(processes=4, initializer=self._init_worker, initargs=(self.tokenizer, self.max_length)) as pool:
            results = list(tqdm(pool.imap(self._worker_func, features), total=len(features), desc="Preprocessing"))

        _all_token_ids, _all_attention_masks, _all_label_ids = zip(*results)
        for idx, value in enumerate(_all_token_ids):
            all_token_ids.append(value)
            all_attention_masks.append(_all_attention_masks[idx])
            all_label_ids.append(_all_label_ids[idx])

        return (
            torch.tensor(all_token_ids, dtype=torch.long),
            torch.tensor(all_attention_masks, dtype=torch.long),
            torch.tensor(all_label_ids, dtype=torch.long),
        )

    @staticmethod
    def _init_worker(tokenizer, max_length):
        global worker_tokenizer, worker_max_len

        worker_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=True, padding_side="left", use_fast=True
        )
        worker_max_len = max_length

    @staticmethod
    def _worker_func(features):
        """
        RETURN:
            - input_ids: tokens
            - attention_mask: A list of length equal to input_ids, where 0 represents a pad and 1 represents a normal token.
            - vaild_len: A list of length equal to input_ids, where 1 represents the valid part (this part requires loss calculation).
        """
        global worker_tokenizer, worker_max_len

        split_marker = "<|im_start|>assistant\n"
        user_part = features.split(split_marker)[0] + split_marker

        full_tokens = worker_tokenizer(features, truncation=True, padding='max_length', max_length=worker_max_len)
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        # 得到label: 将pad和prefix均设置为-100且不参与损失计算，之外将有效文本左移一位保证token对应的lable是token+1
        # 最后的 <|im_end|> 对应lable为-100
        pad_len = 0
        for i in input_ids:
            if i == worker_tokenizer.pad_token_id:
                pad_len += 1
        prefix_tokens = worker_tokenizer(user_part, truncation=False, padding=False, add_special_tokens=False)[
            "input_ids"
        ]
        prefix_len = pad_len + len(prefix_tokens)

        label_ids = [-100] * prefix_len + input_ids[prefix_len + 1 :] + [-100]
        # vaild_len = [0] * prefix_len + [1] * (worker_max_len - prefix_len)

        return input_ids, attention_mask, label_ids


def _get_csv_data():
    csv = pd.read_csv("/root/autodl-tmp/SFTwithLoRA/data/case.csv")
    texts = []

    for title, step, result in zip(csv["用例标题"], csv["测试步骤"], csv["预期结果"]):
        texts.append(
            f"<|im_start|>user\n{title.strip()}<|im_end|>\n<|im_start|>assistant\n测试步骤:\n{step.strip()}\n预期结果:\n{result.strip()}<|im_end|>"
        )
    return texts


def get_dateiter(batch_size: int, max_length: int = 1024, num_workers: int = 0, is_train: bool = True):
    texts = _get_csv_data()
    tokenizer = "/root/autodl-tmp/SFTwithLoRA/LLM"

    dataset = CaseDateset(texts, tokenizer, max_length)  # tokens, vaild_len, label

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    data_iter = get_dateiter(batch_size=10, max_length=192)
