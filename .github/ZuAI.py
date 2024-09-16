import math
import time
import jieba
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torchtext.data import Field, BucketIterator, Example
from torchtext.data import TabularDataset
import numpy as np
import random
from torchtext.data.metrics import bleu_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import nltk.translate.bleu_score as bleu
import wandb
wandb.login()
wandb.init(project='Zu_bot', entity='ZU')

# =============================================================  配置参数  =====================================================================

config = {
    "split_ratio": [0.7, 0.2, 0.1],  # 数据集划分比例
    "batch_size": 185,  # 批处理大小
    "embedding_dim": 256,  # 词嵌入维度
    "nhead": 8,  # 多头注意力头数
    "num_encoder_layers": 5,  # 编码器层数
    "num_decoder_layers": 5,  # 解码器层数
    "dropout": 0.2,  # 随机失活概率
    "temperature":0.07,
    "learning_rate": 0.0001,  # 学习率
    "num_epochs": 180,  # 训练轮数
    "clip": 5,  # 梯度裁剪阈值
    "patience": 5,  # 用于早停的耐心参数
    "beam_size": 20,  # Beam搜索的大小
    "max_length": 25  # 生成句子的最大长度
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================  數據處理  ========================================================================
# 定义分词函数
def tokenize_chinese(text):
    return list(jieba.cut(text))

# 初始化Field
TEXT = Field(tokenize=tokenize_chinese, lower=True, init_token='<sos>', eos_token='<eos>')
fields = [('src', TEXT), ('trg', TEXT)]

# 从AI.txt中提取问题和回复
questions = []
replies = []
lccc = "/kaggle/input/ai-lccc/lccc.txt"

# 从数据集中提取50%的样本(電腦爛)
with open(lccc, "r", encoding="utf-8") as file:
    lines = file.readlines()
    total_lines = len(lines)
    sample_size = int(total_lines * 0.5)  

    sampled_lines = random.sample(lines, sample_size)

    for line in sampled_lines:
        sample = eval(line)  # 将字符串转换为字典

        # 提取对话列表
        dialog = sample.get('dialog', [])

        # 如果对话列表为空，跳过此样本
        if not dialog:
            continue

        # 将对话列表拆分为问题和回答
        for i in range(len(dialog) - 1):
            questions.append(dialog[i])  # 当前对话是问题
            replies.append(dialog[i + 1])  # 下一对话是回答

# 打印一些问题和回答以确保它们被正确读取
for i in range(1):
    print("Question:", questions[i])
    print("Reply:", replies[i])
    print()

# 确保没有None的数据
filtered_qr_pairs = [(q, r) for q, r in tqdm(zip(questions, replies), desc="Filtering Data") if q is not None and r is not None]
print("Filtering completed!")

start_time = time.time()

split_ratio = config["split_ratio"]
# 划分训练集
train_pairs, remaining_pairs = train_test_split(filtered_qr_pairs, test_size=1 - split_ratio[0])

# 计算剩余数据的比例
remaining_ratio = [split_ratio[1] / (split_ratio[1] + split_ratio[2]), split_ratio[2] / (split_ratio[1] + split_ratio[2])]

# 划分验证集和测试集
test_pairs, valid_pairs= train_test_split(remaining_pairs, test_size=remaining_ratio[0])

# 打印每个数据集的大小
print(f"训练集大小: {len(train_pairs)}")
print(f"验证集大小: {len(valid_pairs)}")
print(f"测试集大小: {len(test_pairs)}")

end_time = time.time()
print(f"train_test_split took {end_time - start_time:.2f} seconds")

# 分割成训练、验证和测试集
train_questions, train_replies = zip(*tqdm(train_pairs, desc="Processing train pairs"))
valid_questions, valid_replies = zip(*tqdm(valid_pairs, desc="Processing valid pairs"))
test_questions, test_replies = zip(*tqdm(test_pairs, desc="Processing test pairs"))

# 將 valid_questions 和 valid_replies 分割成多個子列表
valid_batch_size = 1000  # 定義批次大小
question_batches = [valid_questions[i:i+valid_batch_size] for i in range(0, len(valid_questions), valid_batch_size)]
reply_batches = [valid_replies[i:i+valid_batch_size] for i in range(0, len(valid_replies), valid_batch_size)]

# 初始化 valid_examples
valid_examples = []

# 遍歷 question_batches 和 reply_batches
for questions, replies in tqdm(zip(question_batches, reply_batches), desc="Creating Valid Examples"):
    # 對每個批次創建 Examples
    examples = [Example.fromlist([q, r], fields) for q, r in zip(questions, replies)]
    # 將 Examples 添加到 valid_examples
    valid_examples.extend(examples)

# 为TorchText创建Examples
train_examples = [Example.fromlist([q, r], fields) for q, r in tqdm(zip(train_questions, train_replies), desc="Creating Train Examples")]
test_examples = [Example.fromlist([q, r], fields) for q, r in tqdm(zip(test_questions, test_replies), desc="Creating Test Examples")]

# 将Example对象列表写入文件
with open("train_examples.txt", "w", encoding="utf-8") as f:
    for ex in train_examples:
        f.write(f"{ex.src}\t{ex.trg}\n")

# 创建TabularDataset
train_data = TabularDataset(
    path="train_examples.txt",
    format='tsv',  # 使用制表符分隔的文本文件
    fields=[('src', TEXT), ('trg', TEXT)]  # 字段
)

# 将 Example 对象列表写入 csv 文件
with open("valid_examples.txt", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['src', 'trg'])  # 写入标题行
    for ex in valid_examples:
        writer.writerow([ex.src, ex.trg])  # 写入每一行

# 创建 TabularDataset
valid_data = TabularDataset(
    path="valid_examples.txt",
    format='csv',  # 使用逗号分隔的csv文件
    fields=[('src', TEXT), ('trg', TEXT)],  # 字段
    skip_header=True  # 跳过csv文件的标题行
)

# 将 Example 对象列表写入 csv 文件
with open("test_examples.txt", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(['src', 'trg'])  # 写入标题行
    for ex in test_examples:
        writer.writerow([ex.src, ex.trg])  # 写入每一行

# 创建 TabularDataset
test_data = TabularDataset(
    path="test_examples.txt",
    format='csv',  # 使用逗号分隔的csv文件
    fields=[('src', TEXT), ('trg', TEXT)],  # 字段
    skip_header=True  # 跳过csv文件的标题行
)

# 估算训练数据集中的总单词数
total_words = sum(len(example.src) + len(example.trg) for example in train_data.examples)

# 使用tqdm的手动模式来观察建立词汇表的进度
pbar = tqdm(total=total_words, desc="Building Vocabulary")

def update_tqdm(multiplier=1):
    pbar.update(multiplier)

# 在每次添加单词时调用上述函数更新进度条
TEXT.build_vocab(train_data, vectors=None, unk_init=None, min_freq=1, specials_first=False, vectors_cache=None, specials=['<unk>', '<pad>', '<sos>', '<eos>'])

# 创建数据迭代器
train_iterator = BucketIterator(train_data, batch_size=config["batch_size"], sort_key=lambda x: len(x.src), shuffle=True)
valid_iterator = BucketIterator(valid_data, batch_size=config["batch_size"], sort_key=lambda x: len(x.src))
test_iterator = BucketIterator(test_data, batch_size=config["batch_size"], sort_key=lambda x: len(x.src))
# ========================================================  模型定义  =================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵（Positional Encoding Matrix）
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算位置编码的权重
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 填充位置编码矩阵的奇数列和偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在批次维度上添加一个维度并进行转置，以匹配输入维度
        pe = pe.unsqueeze(0).transpose(0, 1)

        # 将位置编码矩阵注册为模型的缓冲区（buffer）
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        # 将位置编码矩阵与输入张量相加，以为输入数据添加位置信息
        x = x + self.pe[:x.size(0), :]

        # 应用丢失（dropout）操作以减少过拟合
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_encoder_layers, num_decoder_layers, dropout):
        super(TransformerModel, self).__init__()

        # 创建嵌入层，将输入序列中的词汇映射为嵌入向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 添加位置编码层以为输入序列添加位置信息
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # 创建Transformer模型，它包括编码器和解码器部分
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dropout=dropout)

        # 创建全连接层，将Transformer输出映射回词汇空间


    def forward(self, src, trg):

        # 对输入序列和目标序列应用嵌入和位置编码
        src_embedded = self.embedding(src) * math.sqrt(config["embedding_dim"])
        src_embedded = self.pos_encoder(src_embedded)

        trg_embedded = self.embedding(trg) * math.sqrt(config["embedding_dim"])
        trg_embedded = self.pos_encoder(trg_embedded)

        # 使用Transformer进行序列到序列的转换
        output = self.transformer(src_embedded, trg_embedded)

        # 将Transformer的输出映射回词汇空间
        output = self.fc(output)
        return output

# 创建Transformer模型的实例
model = TransformerModel(
    len(TEXT.vocab),
    embedding_dim=config["embedding_dim"],
    nhead=config["nhead"],
    num_encoder_layers=config["num_encoder_layers"],
    num_decoder_layers=config["num_decoder_layers"],
    dropout=config["dropout"]
).to(device)

# 定义优化器，用于更新模型的参数
optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

# 定义损失函数，用于计算模型的预测与实际目标之间的差异
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)

# ============================================================  训练和评估  ========================================================

def calculate_bleu(reference, candidate):

    # Tokenize the reference and candidate sentences
    reference_tokenized = [sentence.split() for sentence in reference]
    candidate_tokenized = [sentence.split() for sentence in candidate]
    # Convert numpy int64 to string
    reference_sentences = [str(sentence) for sentence in reference]
    candidate_sentences = [str(sentence) for sentence in candidate]
    # Calculate BLEU score
    bleu_score = bleu.corpus_bleu(reference_tokenized, candidate_tokenized)

    return bleu_score

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for batch in tqdm(iterator, desc="Training"):
        src, trg = batch.src, batch.trg
        
        # 将数据移动到GPU
        src = src.cpu()
        trg = trg.cpu()

        optimizer.zero_grad()
        output = model(src, trg[:-1, :])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = trg[1:].view(-1)
        loss = criterion(output, target)
        loss.backward()
        
        # 清除无用的中间计算图，释放内存
        del output
        del target
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        # 手动释放显存
        torch.cuda.empty_cache()

    return epoch_loss / len(iterator)

# 创建一个学习率调度器，以在验证损失不降低时降低学习率
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    bleu_score = 0

    with torch.no_grad():
        for batch in iterator:
            src, trg = batch.src, batch.trg
            output = model(src, trg)  # Turn off teacher forcing
            output_dim = output.shape[-1]

            # Flatten output and target for calculating loss
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            # Calculate BLEU score
            predicted_sentences = torch.argmax(output, dim=-1).cpu().numpy()
            target_sentences = trg.view(-1, batch.trg.shape[1]).cpu().numpy()
            print(f"predicted_sentences:\n{predicted_sentences}")
            print(f"predicted_sentences:\n{target_sentences}")
            bleu_score += calculate_bleu(predicted_sentences, target_sentences)

    return epoch_loss / len(iterator), bleu_score / len(iterator)


# 从配置中获取训练的总周期数
num_epochs = config["num_epochs"]

# 梯度裁剪阈值
clip = config["clip"]

# 早停的计数器
early_stopping_counter = 0

# 早停的最大容忍周期数
patience = config["patience"]

# 最佳验证损失的初始值（设置为正无穷大）
best_valid_loss = float('inf')

# 保存最佳模型检查点
def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, filename)
    print(f"Saved checkpoint to {filename}")

# 加载模型检查点
def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(f"No checkpoint found at {filename}")

# 循环遍历训练周期
for epoch in range(num_epochs):
    # 训练模型并获取训练损失
    train_loss = train(model, train_iterator, optimizer, criterion, clip)

    # Evaluate model and obtain validation loss and BLEU score
    evaluation_result_valid = evaluate(model, valid_iterator, criterion)

    # Check if evaluation_result is a tuple (loss, bleu)
    if isinstance(evaluation_result_valid, tuple):
        valid_loss, valid_bleu = evaluation_result_valid
        # Log BLEU score for validation set using WandB
        wandb.log({"Train Loss": train_loss, "Valid BLEU": valid_bleu})
        # Log training loss and validation loss using WandB
        wandb.log({"Train Loss": train_loss, "Valid Loss": valid_loss})
    else:
        valid_loss = evaluation_result_valid
        # Log validation loss using WandB
        wandb.log({"Train Loss": train_loss, "Valid Loss": valid_loss})

    # Evaluate model and obtain test BLEU score
    evaluation_result_test = evaluate(model, test_iterator, criterion)

    # Check if evaluation_result is a tuple (loss, bleu)
    if isinstance(evaluation_result_test, tuple):
        test_loss, test_bleu = evaluation_result_test
        # Log BLEU score for test set using WandB
        wandb.log({"Test BLEU": test_bleu})
    else:
        test_loss = evaluation_result_test
        # Log test loss using WandB
        wandb.log({"Test Loss": test_loss})
    # 如果验证损失更好，保存模型检查点
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        early_stopping_counter = 0
        save_checkpoint(model, optimizer, "best_model_checkpoint.pth.tar")
    else:
        # 如果验证损失没有改善，增加早停计数器
        early_stopping_counter += 1
        
        # 如果早停计数达到早停容忍周期，停止训练
        if early_stopping_counter >= patience:
            print("Early Stopping!")
            break



# ============================================================  Beam Search  =============================================================

# Beam search算法
class Beam:
    def __init__(self, log_prob, prev_beam, token_id, tokens):
        self.log_prob = log_prob  # 當前beam的對數概率
        self.prev_beam = prev_beam  # 前一個beam對象
        self.token_id = token_id  # 當前時間步的token索引
        self.tokens = tokens  # 當前beam已生成的token序列

    def get_tokens(self):
        tokens = []
        current_beam = self
        while current_beam:
            tokens.append(current_beam.token_id)
            current_beam = current_beam.prev_beam
        return tokens[::-1]  # 返回當前beam生成的token序列（反向）

    def is_eos(self, eos_token_id):
        return self.token_id == eos_token_id  # 判斷當前時間步的token是否為結束標記

    def __str__(self):
        return f"Beam: log_prob={self.log_prob}, token_id={self.token_id}"  # 返回當前beam的字符串表示形式（用於調試和輸出）

def beam_search(model, sentence, TEXT, device, beam_size=config["beam_size"], max_length=config["max_length"], length_penalty=0.9):
    model.eval()
    model = model.to(device)
    
    init_token_id = TEXT.vocab.stoi[TEXT.init_token]
    eos_token_id = TEXT.vocab.stoi[TEXT.eos_token]

    tokens = [init_token_id] + TEXT.tokenize(sentence)  # 將初始token添加到句子中
    src_indexes = [init_token_id] + [TEXT.vocab.stoi[token] for token in tokens]  # 獲取源語言句子的索引
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)  # 將源語言句子轉換為張量並移動到適當的設備
    beams = [Beam(0, None, init_token_id, [init_token_id])]  # 初始化beam列表，初始beam包含初始token

    for step in range(1, max_length + 1):  # 迭代生成目標語言的每個token
        next_beams = []  # 存儲下一個時間步的beam列表
        all_ended = True  # 指示所有beam是否已經結束

        for beam in beams:  # 對當前時間步的每個beam執行操作
            if beam.is_eos(eos_token_id):  # 如果當前beam已經生成了結束標記，則將其添加到下一個時間步的beam列表中
                next_beams.append(beam)
            else:
                all_ended = False  # 標記為False，表示仍有beam未結束
                trg_tokens = beam.get_tokens()  # 獲取當前beam已生成的目標語言token序列
                trg_tensor = torch.LongTensor(trg_tokens).unsqueeze(1).to(device)  # 將目標語言token序列轉換為張量並移動到適當的設備

                with torch.no_grad():
                    output = model(src_tensor, trg_tensor)  # 使用模型生成下一個目標語言token的概率分佈

                log_probs = torch.log_softmax(output[-1], dim=1)  # 計算下一個token的對數概率
                scaled_log_probs = log_probs / (len(trg_tokens) ** length_penalty)  # 對概率進行懲罰
                topk_log_probs, topk_tokens = scaled_log_probs.topk(beam_size)  # 獲取top-k個概率最高的token及其對應的對數概率

                for k in range(beam_size):  # 對於每個概率最高的token
                    next_token = topk_tokens[0][k].item()  # 獲取token的索引
                    next_tokens = beam.tokens + [next_token]  # 將新token添加到當前beam的token序列中
                    next_beam = Beam(beam.log_prob + topk_log_probs[0][k].item(), beam, next_token, next_tokens)  # 創建新的beam對象
                    next_beams.append(next_beam)  # 將新的beam對象添加到下一個時間步的beam列表中

        next_beams.sort(key=lambda b: -b.log_prob)  # 根據beam的log概率對beam列表進行排序
        beams = next_beams[:beam_size]  # 選擇top-k個log概率最高的beam作為下一個時間步的beam列表

        if all_ended:  # 如果所有beam都已經結束，則停止迭代
            break

    best_beam = beams[0]  # 獲取log概率最高的beam
    best_tokens = best_beam.get_tokens()  # 獲取該beam的token序列
    trg_tokens = [TEXT.vocab.itos[t] for t in best_tokens]  # 將token序列轉換回文本形式
    generated_sentence = ' '.join(trg_tokens)  # 將token序列拼接成生成的句子
    return generated_sentence  # 返回生成的句子

# ============================================================  聊天功能  =============================================================

# 交互式聊天
while True:
    user_input = input("请输入您的消息 (输入 'exit' 退出): ")
    if user_input.lower() == 'exit':
        break
    response = beam_search(model, user_input, TEXT, device)
    print(response)
