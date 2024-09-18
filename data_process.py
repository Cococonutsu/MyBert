import os
import re
import random
import pandas as pd


def get_data(data_path):
    all_content = pd.read_csv(data_path)["content"].tolist()
    return all_content


def cut_sentence(text):
    '''
    该函数用来将一篇文章按照我设定的符号进行切分
    :param text: 每一篇单独的文章
    :return: 文章中每个句子和句子结尾的符号
    '''
    punc_2_index = {"，": 1, "。": 2, "；": 3, "！": 4, "？": 5, "、": 6}
    # 使用()和|设置的re匹配的flatten可以在匹配的同时保留原始符号，如果使用[]的话无法保留句子中的符号
    flatten = "(" + "|".join([key for key in punc_2_index.keys()]) + ")"
    # 符号和句子结合的列表
    split_sentences = []
    # 按照标点符号进行切分
    sentences = re.split(flatten, text)
    # 如果该text以符号结尾会导致最后多出一个空字符串，这里用来消除掉最后一个空字符串
    sentences = sentences[:-1] if sentences[-1] == "" else sentences
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i] + sentences[i + 1]
        split_sentences.append(sentence)
    return split_sentences


def merge_sentence(split_sentences, max_length):
    '''
    由于我们使用cut函数将句子切分，切分后每一小句都很短，这会训练出的模型只能对很短的序列进行预测。
    因此我们设定一个合理的长度，把句子拼接起来。同时还要保留一些短的句子，也可以创造一部分很长的句子以增加样本的多样性。

    此外增加一定几率的特殊处理：
    - 10% 几率直接将 sentence 存入 merge_sentences【短样本】
    - 10% 几率允许超过 max_length 继续拼接【长样本】
    - 80% 几率按正常逻辑处理【普遍样本】

    :param split_sentences: 使用cut_sentences切分好的列表
    :param max_length: 每个拼接句子的最大长度
    :return: 拼接到指定长度的句子
    '''
    merge_sentences = []
    result = ""
    for sentence in split_sentences:
        rand_num = random.random()
        if rand_num < 0.1:
            merge_sentences.append(sentence)
            continue

        elif rand_num < 0.2:
            result += sentence
            continue

        if len(result) + len(sentence) > max_length:
            merge_sentences.append(result)
            result = sentence
        else:
            result += sentence

    if result:
        merge_sentences.append(result)
    return merge_sentences


def build_every_text_dataset(merge_sentences):
    '''
    对每一个文章进行数据集的创建
    text_1表示的一句话
    text_2表示第二句话
    label表示两句话是否相邻，1表示相邻局，0表示非相邻句
    :param merge_sentences:
    :return:
    '''
    text_1 = []
    text_2 = []
    label = []
    if len(merge_sentences) <= 2:
        return text_1, text_2, label
    for idx in range(0, len(merge_sentences) - 1):
        if len(merge_sentences[idx]) == 0 or len(merge_sentences[idx+1]) == 0:
            continue
        text_1.append(merge_sentences[idx])
        text_2.append(merge_sentences[idx + 1])
        label.append(1)

        rand_id = random.choice([i for i in range(len(merge_sentences)) if i != idx and i != idx + 1])
        if len(merge_sentences[idx]) == 0 or len(merge_sentences[rand_id]) == 0:
            continue
        text_1.append(merge_sentences[idx])
        text_2.append(merge_sentences[rand_id])
        label.append(0)

    return text_1, text_2, label


def func(all_content, task2_dataset):
    '''
    创建完整的数据集
    :param all_content:
    :return:
    '''
    all_data_text_1 = []
    all_data_text_2 = []
    all_data_label = []
    for text in all_content:
        split_sentences = cut_sentence(text)
        merge_sentences = merge_sentence(split_sentences, 45)
        text_1, text_2, label = build_every_text_dataset(merge_sentences)
        all_data_text_1.extend(text_1)
        all_data_text_2.extend(text_2)
        all_data_label.extend(label)
    pd.DataFrame({"text1": all_data_text_1, "text2": all_data_text_2, "label": all_data_label}).to_csv(
        task2_dataset, index=False)


def build_vocab(all_content, vocab_txt):
    if os.path.exists(vocab_txt):
        with open(vocab_txt, "r", encoding="utf-8") as fp:
            ##########################################################
            # 这里原来读取index_2_word的代码是,但是按照下面的读取方式会缺少数据
            # index_2_word = [line.strip() for line in fp.readlines()]
            # 该问题还没有完全解决
            ##########################################################
            index_2_vocab = fp.read().split("\n")
            if index_2_vocab[-1] == '':  # 移除空行
                index_2_vocab = index_2_vocab[:-1]
            vocab_2_index = {vocab: index for index, vocab in enumerate(index_2_vocab)}
            return vocab_2_index, index_2_vocab

    vocab_2_index = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4}

    for text in all_content:
        for word in text:
            if word not in vocab_2_index:
                vocab_2_index[word] = len(vocab_2_index)

    index_2_vocab = list(vocab_2_index)

    with open(vocab_txt, "w", encoding="utf-8") as fp:
        #####################################################
        # 同样使用下面这种方式写入数据不会出现最后一行是"\n"
                fp.write("\n".join(index_2_vocab))
    return vocab_2_index, index_2_vocab


if __name__ == "__main__":
    task2_dataset = "./data/task2_dataset.csv"
    vocab_txt = "./data/vocab.txt"

    all_content = get_data("data/unlabeled_data.csv")

    # 检查数据集是否存在，如不存在则创建数据集
    if not os.path.exists(task2_dataset):
        func(all_content, task2_dataset)

    vocab_2_index, index_2_vocab = build_vocab(all_content, vocab_txt)