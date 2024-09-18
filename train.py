import os
import torch
import random
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Bert_encoder import Bert_Encoder, Cls_Attention


def load_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    all_text_1 = df["text1"].tolist()
    all_text_2 = df["text2"].tolist()
    all_label = df["label"].tolist()
    return all_text_1, all_text_2, all_label


def load_vocab(vocab_txt_path):
    with open(vocab_txt_path, "r", encoding="utf-8") as fp:
        ##########################################################
        # 这里原来读取index_2_word的代码是,但是按照下面的读取方式会缺少数据
        # index_2_word = [line.strip() for line in fp.readlines()]
        # 该问题还没有完全解决
        ##########################################################
        index_2_word = fp.read().split("\n")
        word_2_index = {vocab: index for index, vocab in enumerate(index_2_word)}
        return word_2_index, index_2_word


class BDdataset(Dataset):
    def __init__(self, all_text_1, all_text_2, all_label, max_len, word_2_index):
        super().__init__()
        self.all_text_1 = all_text_1
        self.all_text_2 = all_text_2
        self.all_label = all_label
        self.max_len = max_len
        self.word_2_index = word_2_index

        assert len(all_text_1) == len(all_text_2) == len(all_label), "text1-text2-label三者长度不相等"

    def __getitem__(self, index):
        text_1 = self.all_text_1[index]
        text_2 = self.all_text_2[index]
        label = self.all_label[index]

        # 为了确保句子的长度不超过我设定的max_len这里对每一个句子进行截断
        # 【截断的策略可以自定义，这里选择各自保留最大长度的一半】减去2是因为要给后面的[CLS][SEP][SEP]留位置
        text_1_idx = [self.word_2_index.get(vocab, word_2_index["[UNK]"]) for vocab in text_1][:(self.max_len // 2) - 2]
        text_2_idx = [self.word_2_index.get(vocab, word_2_index["[UNK]"]) for vocab in text_2][:(self.max_len // 2) - 2]

        text_idx = [word_2_index["[CLS]"]] + text_1_idx + [word_2_index["[SEP]"]] + text_2_idx + [word_2_index["[SEP]"]]
        PAD_num = self.max_len - len(text_idx)
        text_idx = text_idx + [word_2_index["[PAD]"]] * PAD_num

        seg_idx = [0] + [0] * len(text_1_idx) + [0] + [1] * len(text_2_idx) + [1] + [2] * PAD_num

        mask_val = [0 for i in range(len(text_idx))]
        for i, value in enumerate(text_idx):
            if value in [word_2_index["[CLS]"], word_2_index["[SEP]"], word_2_index["[PAD]"]]:
                continue
            else:
                rand_num = random.random()
                if rand_num <= 0.15:
                    rand_num = random.random()
                    if rand_num <= 0.8:
                        text_idx[i] = word_2_index["[MASK]"]
                        mask_val[i] = value
                    elif rand_num >= 0.9:
                        rand_index = random.randint(5, len(word_2_index) - 1)
                        text_idx[i] = rand_index
                        mask_val[i] = value

        return torch.tensor(text_idx), torch.tensor(label), torch.tensor(mask_val), torch.tensor(seg_idx)

    def __len__(self):
        return len(self.all_text_1)


class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.pos_embedding = nn.Embedding(config["max_pos_embedding"], config["hidden_size"])
        self.seg_embedding = nn.Embedding(config["type_vocab_size"], config["hidden_size"])
        self.layernorm = nn.LayerNorm(config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout_prob"])

    def forward(self, batch_text_idx, batch_seg_idx):
        word_emb = self.word_embedding(batch_text_idx)
        pos_idx = torch.arange(0, config["max_pos_embedding"], device=config["device"])

        ######################################################################################################
        # 因为pos_idx不是通过接收数据集的数据来创建的，而是每一个batch进入模型的时候创建pos_idx                           #
        # 因此要保持pos_idx第一个batch维度和传进来的batch_text_idx和batch_seg_idx的batch维度相等                     #
        # 原来传递的参数是config["batch_size"]，但是由于这个参数是固定的                                             #
        # 因此如果最后一个batch中数据的长度小于config["batch_size"]的话会导致pos_emb的第一个维度和word_emb和seg_emb     #
        # 的第一个维度不相等从导致而三者无法相加                                                                    #
        #####################################################################################################
        pos_idx = pos_idx.repeat(batch_text_idx.shape[0], 1)

        pos_emb = self.pos_embedding(pos_idx)
        seg_emb = self.seg_embedding(batch_seg_idx)
        emb = word_emb + pos_emb + seg_emb
        emb = self.layernorm(emb)
        emb = self.dropout(emb)
        return emb


class Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dense = Cls_Attention(config["hidden_size"], config["head_num"])
        self.activation = nn.Tanh()

    def forward(self, x):
        first_token_tensor = x[:, 0]
        out = self.dense(first_token_tensor)
        out = self.activation(out)
        return out


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = BertEmbedding(config)
        self.bert_layer = nn.Sequential(
            *[Bert_Encoder(config["hidden_size"], config["feed_num"], config["head_num"]) for i in
            range(config["Encoder_num"])])
        # self.bert_layer = nn.Linear(config["hidden_size"], config["hidden_size"]) 简易Bert_ayer
        self.pooler = Pooler(config)

    def forward(self, batch_text_idx, batch_seg_idx):
        emb_out = self.embedding(batch_text_idx, batch_seg_idx)
        bert_out_1 = self.bert_layer(emb_out)
        bert_out_2 = self.pooler(bert_out_1)
        return bert_out_1, bert_out_2


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)
        self.mask_cls = nn.Linear(config["hidden_size"], config["vocab_size"])
        self.nsp_cls = nn.Linear(config["hidden_size"], 2)

        self.mask_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.next_loss = nn.CrossEntropyLoss()

    def forward(self, batch_text_idx, batch_seg_idx, batch_label=None, batch_mask_val=None):
        bert_out_1, bert_out_2 = self.bert(batch_text_idx, batch_seg_idx)

        pre_mask = self.mask_cls(bert_out_1)
        pre_cls = self.nsp_cls(bert_out_2)

        if batch_mask_val is None and batch_label is None:
            return torch.argmax(pre_mask, dim=-1), torch.argmax(pre_cls, dim=-1)

        mask_loss = self.mask_loss(pre_mask.reshape(-1, pre_mask.shape[-1]), batch_mask_val.reshape(-1))
        pre_loss = self.next_loss(pre_cls, batch_label)
        loss = mask_loss + pre_loss

        return loss


if __name__ == "__main__":
    result_folder = './result'
    if not os.path.exists(result_folder):
        # 如果文件夹不存在，则创建
        os.makedirs(result_folder)
    # 初始化 result_folder 里的文件
    train_loss_path = os.path.join(result_folder, "train_loss.txt")
    mask_acc_path = os.path.join(result_folder, "mask_acc.txt")
    cls_acc_path = os.path.join(result_folder, "cls_acc.txt")

    # 如果文件存在，先删除它们
    if os.path.exists(train_loss_path):
        os.remove(train_loss_path)
    if os.path.exists(mask_acc_path):
        os.remove(mask_acc_path)
    if os.path.exists(cls_acc_path):
        os.remove(cls_acc_path)

    # -------------------加载词汇表----------------------
    vocab_txt = "./data/vocab.txt"
    word_2_index, index_2_word = load_vocab(vocab_txt)

    # -------------------加载数据集----------------------
    dataset_path = "./data/task2_dataset.csv"
    all_text_1, all_text_2, all_label = load_dataset(dataset_path)

    # -------------------训练所需全部参数----------------------
    config = {
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "epochs": 10,
        "batch_size": 40,
        "shuffle": True,
        "hidden_size": 768,
        "feed_num": 1024,
        "head_num": 4,
        "Encoder_num": 3,
        "lr": 1e-3,
        "vocab_size": len(word_2_index),
        "max_pos_embedding": 128,
        "type_vocab_size": 3,
        "dropout_prob": 0.2,
        "test_size": 10000,
    }
    # -------------------创建训练集和测试集----------------------
    train_dataset = BDdataset(all_text_1[:-config["test_size"]], all_text_2[:-config["test_size"]],
                            all_label[:-config["test_size"]], config["max_pos_embedding"],
                            word_2_index)
    train_dataloader = DataLoader(train_dataset, config["batch_size"], config["shuffle"])

    test_dataset = BDdataset(all_text_1[-config["test_size"]:], all_text_2[-config["test_size"]:],
                            all_label[-config["test_size"]:], config["max_pos_embedding"],
                            word_2_index)
    test_dataloader = DataLoader(test_dataset, config["batch_size"])

    # -------------------定义模型----------------------
    model = Model(config).to(config["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # -------------------收集数据----------------------
    all_loss_list = []
    all_mask_acc = []
    all_cls_acc = []

    # -------------------训练and测试----------------------
    print("devoce:{}".format(config["device"]))
    for epoch in range(config["epochs"]):
        print(f"=====================================Epoch:[{epoch}]=====================================")

        for train_i, train_batch in enumerate(train_dataloader):

            train_batch = [tensor.to(config["device"]) for tensor in train_batch]
            batch_text_idx, batch_label, batch_mask_val, batch_seg_idx = train_batch

            loss = model(batch_text_idx, batch_seg_idx, batch_label, batch_mask_val)
            loss.backward()

            opt.step()
            opt.zero_grad()

            # 使用 tqdm.write 在不干扰进度条的情况下输出训练轮次和损失
            if train_i % 100 == 0:
                with open(train_loss_path, "a", encoding="utf-8") as fp:
                    fp.write(f"{round(loss.item(), 3)}\n")
                all_loss_list.append(round(loss.item(), 3))
                print(f"Train {train_i} Loss:[{loss.item():.3f}]")

            if train_i % 300 == 0:
                model.eval()

                mask_right_num = 0
                mask_all_num = 0

                cls_right_num = 0
                cls_all_num = 0

                for test_i, test_batch in enumerate(test_dataloader):
                    test_batch = [tensor.to(config["device"]) for tensor in test_batch]
                    test_text_idx, test_label, test_mask_val, test_seg_idx = test_batch

                    pre_mask, pre_cls = model(test_text_idx, test_seg_idx)
                    mask_right_num += int(torch.sum(pre_mask[test_mask_val != 0] == test_mask_val[test_mask_val != 0]))
                    mask_all_num += len(pre_mask[test_mask_val != 0])

                    cls_right_num += int(torch.sum(pre_cls == test_label))
                    cls_all_num += len(pre_cls)

                all_mask_acc.append(mask_right_num / mask_all_num * 100)
                all_cls_acc.append(cls_right_num / cls_all_num * 100)

                with open(mask_acc_path, "a", encoding="utf-8") as fp:
                    fp.write(f"{mask_right_num / mask_all_num * 100}\n")
                with open(cls_acc_path, "a", encoding="utf-8") as fp:
                    fp.write(f"{cls_right_num / cls_all_num * 100}\n")

                print("===============test===============")
                print(f"MASK Acc:【{mask_right_num / mask_all_num * 100}%】")
                print(f"CLS Acc:【{cls_right_num / cls_all_num * 100}%】")
                print("===============test===============")

                model.train()
