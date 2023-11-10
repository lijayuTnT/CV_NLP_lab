import pandas as pd 
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset , random_split
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler

def data_loader(train_file_path,test_file_path,bert_path,name):
    #读取csv文件，因为WNLI和MRPC数据集的格式不同，所以要根据name来区分
    if name=="WNLI":
        df_train = pd.read_csv(train_file_path,delimiter='\t',header=None,names=['sentence0','sentence1','label'])
        df_test =  pd.read_csv(test_file_path,delimiter='\t',header=None,names=['sentence0','sentence1','label'])
        # print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
        # df_train.sample(10)   
    else:
        df_train = pd.read_csv(train_file_path,delimiter='\t',header=None,names=['label','ID1','ID2','sentence0','sentence1'])
        df_test =  pd.read_csv(test_file_path,delimiter='\t',header=None,names=['label','ID1','ID2','sentence0','sentence1'])
        # print('Number of training sentences: {:,}\n'.format(df_train.shape[0]))
        # df_train.sample(10)   

    #因为两个数据集都是为了测试两个句子间的关系，所以合并两个句子方便处理，分为训练部分和测试部分
    sentences_train = df_train.sentence0.values[1:]+df_train.sentence1.values[1:]
    labels_train = df_train.label.values[1:]

    sentences_test =df_test.sentence0.values[1:] + df_test.sentence1.values[1:]
    labels_test = df_test.label.values[1:]

    #获得分词器
    tokenizer = BertTokenizer.from_pretrained(bert_path,do_lower_case = True)

    #max_len是句子的最大长度，超过要裁断，少于要padding
    if name=="WNLI":
        max_len = 128
    else:
        max_len = 512

    #函数 tokenizer.encode_plus 包含以下步骤：
    # 将句子分词为 tokens。
    # 在两端添加特殊符号 [CLS] 和[SEP]。
    # 将 tokens 映射为下标 IDs。
    # 将列表填充或截断为固定的长度。
    # 创建 attention masks，将填充的和非填充 tokens 区分开来
    # 保存好input_ids和attention_masks

    #训练集
    input_ids = []
    attention_masks =[]
    for sent in sentences_train:
        encoded_dict = tokenizer.encode_plus(
                                            sent,
                                            add_special_tokens = True ,#add 'CLS'
                                            max_length = max_len, 
                                            pad_to_max_length = True,   #pad & truncate all sentences to max_length
                                            return_attention_mask =True,  #construct attn. masks
                                            return_tensors = 'pt',     #return pytorch tensors
                                            truncation = True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    #将列表变为tensor形式方便使用
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_train = [int(label) for label in labels_train]
    labels_train = torch.tensor(labels_train)
    # print('Original: ', sentences[0])
    # print('Token IDs:', input_ids[0])

    #构造数据集
    dataset = TensorDataset(input_ids,attention_masks,labels_train)

    #测试集
    input_ids = []
    attention_masks =[]
    for sent in sentences_test:
        encoded_dict = tokenizer.encode_plus(
                                            sent,
                                            add_special_tokens = True ,#add 'CLS'
                                            max_length = max_len, 
                                            pad_to_max_length = True,   #pad & truncate all sentences to max_length
                                            return_attention_mask =True,  #construct attn. masks
                                            return_tensors = 'pt',     #return pytorch tensors
                                            truncation = True
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    #和上面一样转为tensor形式
    labels_test = [int(label) for label in labels_test]
    labels_test = torch.tensor(labels_test)
    testset = TensorDataset(input_ids,attention_masks,labels_test)

    #为了得到验证集，将训练集分割成两部分，训练部分占90%，验证部分占10%
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset , val_dataset =random_split(dataset,[train_size,val_size])

    # print('{:>5,} training samples'.format(train_size))
    # print('{:>5,} validation samples'.format(val_size))
    batch_size = 32

    #调用DataLoader函数获得，训练集，测试集和验证集
    train_dataloader = DataLoader(train_dataset,
                                sampler= RandomSampler(train_dataset),
                                batch_size=batch_size)

    validation_dataloader = DataLoader(val_dataset,
                                    sampler=RandomSampler(val_dataset),
                                    batch_size=batch_size)
    
    prediction_dataloader = DataLoader(testset,
                                    sampler=RandomSampler(testset),
                                    batch_size=batch_size)
    return train_dataloader,validation_dataloader,prediction_dataloader
    