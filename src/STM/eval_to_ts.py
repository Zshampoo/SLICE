# predict_gpt_to_ts.py
import ast
import csv
import os
import tempfile

from src.CheXbert.src.label import label
from src.LLM.CE_calc import clean_report_mimic_cxr, calc_NLG_CE
from src.STM.chexbert_utils import get_chexbert

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
# from src.config import Config
from torch.utils.data import Dataset, DataLoader
import tqdm
device = os.environ.get("DEVICE", "cuda:0" if __import__("torch").cuda.is_available() else "cpu")

SCORES_NP = {'Negative': '[NEG]', 'Positive': '[POS]'}

label_cols = ['No Finding',
              'Enlarged Cardiomediastinum',
              'Cardiomegaly',
              'Lung Opacity',
              'Lung Lesion',
              'Edema',
              'Consolidation',
              'Pneumonia',
              'Atelectasis',
              'Pneumothorax',
              'Pleural Effusion',
              'Pleural Other',
              'Fracture',
              'Support Devices']

def load_model(save_dir, device='cpu'):
    """Load a saved prompted-transfer checkpoint (HuggingFace layout)."""
    return load_trained_model(save_dir)[0], load_trained_model(save_dir)[1], {}


def load_trained_model(dir):
    """加载训练好的模型和分词器"""
    tokenizer = BartTokenizer.from_pretrained(dir)  # 使用训练时保存的tokenizer
    model = BartForConditionalGeneration.from_pretrained(dir)  # 使用训练时保存的模型
    model.to(device)  # 将模型移至GPU
    model.eval()  # 设置为评估模式
    return model, tokenizer

class TextDataset_PROMPT_prompt_BOS(Dataset):
    """自定义数据集类"""
    def __init__(self, texts, tokenizer, max_length=512, prompt=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        inputs_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')

        prompt = self.prompt[idx]
        prompt_text = ' '.join(prompt) + ' '
        prompt_new = '[PROMPT] ' + prompt_text + self.tokenizer.bos_token

        decoder_encoding =  self.tokenizer(prompt_new, max_length=self.max_length,  truncation=True, return_tensors="pt", add_special_tokens=False)

        # prompt_input_ids = prompt_token.input_ids
        # prompt_attention_mask = prompt_token.attention_mask
        # prompt_input_ids[:, 0] = self.tokenizer.bos_token_id
        # prompt_input_ids = prompt_input_ids[:, :-1]
        # prompt_attention_mask = prompt_attention_mask[:, :-1]
        return {
                'input_ids': inputs_encoding['input_ids'].squeeze(),
                'attention_mask': inputs_encoding['attention_mask'].squeeze(),
                'prompt_input_ids': decoder_encoding['input_ids'].squeeze(),
                'prompt_attention_mask':decoder_encoding['attention_mask'].squeeze(),
                'prompt': prompt_new,
            }

class TextDataset_wchexbertprompt(Dataset):
    """自定义数据集类"""
    def __init__(self, texts, tokenizer, max_length=512, prompt=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_report_mimic_cxr(self.texts[idx])
        # 添加prompt
        prompt = self.prompt[idx]
        prompt = ' '.join(prompt) + ' '
        text_new = prompt + text
        # print(text_new)
        inputs = self.tokenizer(text_new, return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding='max_length')

        # prompt_token =  self.tokenizer(prompt, return_tensors="pt")
        # prompt_input_ids = prompt_token.input_ids
        # prompt_attention_mask = prompt_token.attention_mask
        # prompt_input_ids[:, 0] = self.tokenizer.bos_token_id
        # prompt_input_ids = prompt_input_ids[:, :-1]
        # prompt_attention_mask = prompt_attention_mask[:, :-1]
        return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'text_new':text_new
            }


class TextDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, texts, tokenizer, max_length=512, w_prompt = False, labels = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_prompt = w_prompt
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length,
                              truncation=True, padding='max_length')
        return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze()
            }


def clean_wo_np(report):
    report = report.replace('[POS] ','')
    report = report.replace('[NEG] ', '')
    report = report.replace('[PROMPT] ', '')
    return report

def predict_batch(model, batch, tokenizer):
    """批量预测函数"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    if 'text_new' in batch:
        text_new = batch['text_new']
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=4,  # 使用beam search
            length_penalty=1.0
        )

    
    predictions = []
    for i, output in enumerate(outputs):
        pred = tokenizer.decode(output, skip_special_tokens=True)
        predictions.append(pred)
    return predictions

def predict_batch_wprompt(model, batch, tokenizer):
    """批量预测函数"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    prompt_input_ids = batch['prompt_input_ids'].to(device)
    prompt_attention_mask = batch['prompt_attention_mask'].to(device)
    prompt = batch['prompt']

    with torch.no_grad():
        # outputs = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_length=512,
        #     num_beams=4,  # 使用beam search
        #     length_penalty=1.0
        # )
        # print("input_ids max:", input_ids.max().item(), "prompt_input_ids_max:", prompt_input_ids.max().item(), "vocab_size:", model.bart.config.vocab_size)
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=prompt_input_ids,
            decoder_attention_mask=prompt_attention_mask,
            # max_length=512,
            num_beams=4,  # 使用beam search
            length_penalty=1.0,
            repetition_penalty=1.0, # 抑制模型在生成时重复同样的 token 或片段，* 1.0	不使用惩罚机制 1.1 - 1.5	常见取值范围，适度抑制重复 > 2.0	非常强烈惩罚，可能影响生成质量
            # min_length=100,  # 4.25 Transformers
            max_new_tokens=150,
            eos_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    predictions = []
    for i, output in enumerate(outputs):
        pred = tokenizer.decode(output, skip_special_tokens=True)
        bos_str = ' <s> '
        length = len(prompt[i])-len(bos_str)
        pred_woprompt = clean_wo_np(pred[length:])

        # print(f'prompt length: {len(prompt[i])}')
        predictions.append(pred_woprompt)
    return predictions

def predict_and_save(csv_input_path, csv_output_path, checkpoint_dir, llmpred_col, transfer_col, batch_size=32):
    """读取CSV文件，预测gpt列并保存结果到ts列"""
    # 读取原始CSV文件
    if csv_input_path.endswith('.csv'):
        df = pd.read_csv(csv_input_path)
    elif csv_input_path.endswith('.xlsx'):
        df = pd.read_excel(csv_input_path, engine='openpyxl')

    # df = pd.read_excel(csv_input_path, engine="openpyxl")
    print("Loading csv.. from {}".format(csv_input_path))
    # df = pd.read_excel(csv_input_path, engine='openpyxl')
    # 加载训练好的模型
    print("Loading trained model.. from {}".format(checkpoint_dir))
    model, tokenizer = load_trained_model(checkpoint_dir)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")
    print(f"Number of params: {n_parameters:,} ({n_parameters / 1e6:.2f}M)")
    # tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    # tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])
    # model.resize_token_embeddings(len(tokenizer))
    # 创建数据集和数据加载器
    dataset = TextDataset(df[llmpred_col].tolist(), tokenizer)
    # dataset = TextDataset(df['gpt_output_label_clean'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 批量预测
    all_predictions = []
    print("Generating predictions...")
    for batch in tqdm.tqdm(dataloader):
        predictions = predict_batch(model, batch, tokenizer)
        all_predictions.extend(predictions)

    # 将预测结果保存到DataFrame
    # df['transfer_output_pred'] = all_predictions
    # df['transfer_output_pred_clean'] = df['transfer_output'].apply(clean_report_mimic_cxr)

    df[transfer_col] = all_predictions
    df[transfer_col + '_clean'] = df[transfer_col].apply(clean_report_mimic_cxr)
    # 保存新的CSV文件
    # df.to_excel(csv_output_path, engine= 'openpyxl', index=False)
    df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to {csv_output_path}")

def predict_and_save_wprompt(csv_input_path, csv_output_path, checkpoint_dir, llmpred_col, transfer_col, baseline = 'CMCRL', batch_size=16):
    """读取CSV文件，预测gpt列并保存结果到ts列"""
    # 读取原始CSV文件
    df = pd.read_csv(csv_input_path)
    print("Loading CSV from {}, len: {}".format(csv_input_path, len(df)))
    # df = df[:100] # debug
    # 获取chexbert标注得到的prompt
    pred_reports = df[f'llm_infer_wlabelinfo_{baseline}'].tolist()
    pred_reports_clean = [clean_report_mimic_cxr(x) for x in pred_reports]
    input_labels = label_input_report(pred_reports_clean)

    # 按训练时的标签顺序重组prompt
    labels_new, prompt_all = rearrange_labels(input_labels, label_cols)

    # 加载训练好的模型
    print("Loading trained model from {}".format(checkpoint_dir))
    model, tokenizer = load_trained_model(checkpoint_dir)
    print("tokenizer length:",len(tokenizer), "vocab_size:", model.config.vocab_size)
    toks = ['[POS]', '[NEG]','[PROMPT]'] # debug
    for t in toks:
        if t in tokenizer.get_vocab():
            print(f"Token {t} 存在于词表中。")

    # 创建数据集和数据加载器
    dataset = TextDataset_PROMPT_prompt_BOS(texts=df[llmpred_col].tolist(),
                          tokenizer=tokenizer,
                          max_length=512,
                          prompt=prompt_all)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 批量预测
    all_predictions = []
    print("Generating predictions...")
    for batch in tqdm.tqdm(dataloader):
        predictions = predict_batch_wprompt(model, batch, tokenizer)
        all_predictions.extend(predictions)

    df[transfer_col] = all_predictions
    df[transfer_col + '_clean'] = df[transfer_col].apply(clean_report_mimic_cxr)
    # 保存新的CSV文件
    # df.to_excel(csv_output_path, engine= 'openpyxl', index=False)
    df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to {csv_output_path}")

def rearrange_labels(labels, label_cols):
    labels_new = []
    prompt_all = []
    for patient_label in labels:
        patient_label_new = []
        patient_prompt = []
        for lab in label_cols:
            for plab in patient_label:
                if plab[0] == lab:
                    patient_label_new.append(plab)
                    patient_prompt.append(plab[1])
        labels_new.append(patient_label_new)
        prompt_all.append(patient_prompt)
    return labels_new, prompt_all



def predict_and_save_wchexbertinputprompt(csv_input_path, csv_output_path, checkpoint_dir, llmpred_col, transfer_col, batch_size=16, prompt_col = None):
    """读取CSV文件，预测gpt列并保存结果到ts列"""
    # 读取原始CSV文件
    if csv_input_path.endswith('.csv'):
        df = pd.read_csv(csv_input_path)
    elif csv_input_path.endswith('.xlsx'):
        df = pd.read_excel(csv_input_path, engine='openpyxl')

    print("Loading CSV from {}, len: {}".format(csv_input_path, len(df)))
    # df = df[:100]
    # 获取chexbert标注得到的prompt
    # pred_reports = df[f'llm_infer_wlabelinfo_{baseline}'].tolist()
    if prompt_col is None:
        pred_reports = df[llmpred_col].tolist()
        pred_reports_clean = [clean_report_mimic_cxr(x) for x in pred_reports]
        input_labels = label_input_report(pred_reports_clean)

        # 按训练时的标签顺序重组prompt
        labels_new, prompt_all = rearrange_labels(input_labels, label_cols)
    else:
        df[prompt_col] =  df[prompt_col].apply(ast.literal_eval)
        prompt_all = df[prompt_col].values.tolist()
        # print(prompt_all)
    # print(prompt_all)
    # df = pd.read_excel(csv_input_path, engine="openpyxl")

    # df = pd.read_excel(csv_input_path, engine='openpyxl')
    # 加载训练好的模型
    print("Loading trained model from {}".format(checkpoint_dir))
    # model, tokenizer, config = load_model(checkpoint_dir, device)
    model, tokenizer = load_trained_model(checkpoint_dir)
    print("tokenizer length:",len(tokenizer), "vocab_size:", model.config.vocab_size)
    toks = ['[POS]', '[NEG]','[DEC]']
    for t in toks:
        if t in tokenizer.get_vocab():
            print(f"Token {t} 存在于词表中。")
    # tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    # tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])
    # model.resize_token_embeddings(len(tokenizer))
    # 创建数据集和数据加载器
    dataset = TextDataset_wchexbertprompt(texts=df[llmpred_col].tolist(),
                          tokenizer=tokenizer,
                          max_length=512,
                          prompt=prompt_all)
    # dataset = TextDataset(df['gpt_output_label_clean'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 批量预测
    all_predictions = []
    print("Generating predictions...")
    for batch in tqdm.tqdm(dataloader):
        predictions = predict_batch(model, batch, tokenizer)
        all_predictions.extend(predictions)
        # break

    # 将预测结果保存到DataFrame
    # df['transfer_output_pred'] = all_predictions
    # df['transfer_output_pred_clean'] = df['transfer_output'].apply(clean_report_mimic_cxr)

    df[transfer_col] = all_predictions
    df.to_csv(csv_output_path, index=False)
    df[transfer_col + '_clean'] = df[transfer_col].apply(clean_report_mimic_cxr)
    # 保存新的CSV文件
    # df.to_excel(csv_output_path, engine= 'openpyxl', index=False)
    df.to_csv(csv_output_path, index=False)
    print(f"Predictions saved to {csv_output_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DST prompted transfer inference helper")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--llmpred_col", default="llm_report")
    parser.add_argument("--transfer_col", default="dst_report")
    parser.add_argument("--prompt_col", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    predict_and_save_wchexbertinputprompt(
        csv_input_path=args.input_csv,
        csv_output_path=args.output_csv,
        checkpoint_dir=args.checkpoint_dir,
        llmpred_col=args.llmpred_col,
        transfer_col=args.transfer_col,
        batch_size=args.batch_size,
        prompt_col=args.prompt_col,
    )
