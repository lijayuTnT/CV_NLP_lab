from transformers import BertForSequenceClassification , AdamW,BertConfig
import torch
from transformers import get_linear_schedule_with_warmup
from Data_loader import data_loader
from my_utils import flat_accuracy
from tqdm import tqdm

#训练函数，参数name为数据集的名称，bert_path为bert的路径
def train(name , bert_path):
    #依据数据集名称,指定存储模型参数的路径，并调用data_loader函数，获得训练集，验证集和测试集
    if name=="WNLI":
        PATH = "./wnli_finetuned/wnli_model.pth"
        train_dataloader,validation_dataloader,prediction_dataloader = data_loader("data/WNLI/train.tsv","data/WNLI/dev.tsv",bert_path,name)
    else:
        PATH = "./mrpc_finetuned/mrpc_model.pth"
        train_dataloader,validation_dataloader,prediction_dataloader = data_loader("data/MRPC/train.tsv","data/MRPC/dev.tsv",bert_path,name)
   
    #指定设备，方便使用GPU
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(device)
    
    #载入模型（本地载入）
    config = BertConfig.from_pretrained(bert_path)
    model = BertForSequenceClassification.from_pretrained(bert_path, 
                                                        config=config,
                                                        
                                                    )
    model.to(device)

    #优化器使用AdamW，lr的推荐值为5e-5, 3e-5, 2e-5
    optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8)  

    #总训练次数为2，epochs的推荐值为2，3，4，实验过程中发现3和4会过拟合，2最合适      
    epochs = 2

    patience = 5
    total_steps = len(train_dataloader) * epochs

    #调度学习率
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
    stale = 0
    best_acc = 0.0
    print("Start Training")
    for epoch in range(epochs):
        model.train()
        #方便保存每次epoch的损失和准确率
        train_loss = []
        train_accs = []
        for batch in tqdm(train_dataloader):
          
            # input_ids 通过一个输入被分词后每个词在字典里的序号构成的向量组成，padding部分的维度的值为0
            input_ids = batch[0].to(device)
            
            # input_mask 通过的维度和input_ids一样，如果是padding则置为0，如果是分词的序号部分则置为1
            # 掩码会告诉 BERT 中的 “Self-Attention” 机制不去处理这些填充的符号。
            input_mask = batch[1].to(device)

            # 输入的标签
            labels = batch[2].to(device)
            model.zero_grad()

            #调用bert模型
            result = model(input_ids, 
                       token_type_ids=None, 
                       attention_mask=input_mask, 
                       labels=labels,
                       return_dict=True)
            
            #由于输入了labels，所以我们直接调用返回的loss和logits
            loss = result.loss
            logits = result.logits
            loss.backward()
            #梯度裁剪防止“exploding gradients”problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            #flat_accuracy计算准确率
            acc = flat_accuracy(logits , label_ids)
            train_loss.append(loss.item())
            train_accs.append(acc)
        #计算loss和acc的均值
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        #打印训练一步的结果
        print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        
        model.eval()
        valid_loss = []
        valid_accs = []

        #验证部分，和训练类似，不需要调节参数
        for batch in tqdm(validation_dataloader):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device) 
            with torch.no_grad():
                result = model(input_ids,
                                token_type_ids=None, 
                                attention_mask=input_mask,
                                labels=labels,
                                return_dict=True)
                loss = result.loss
                logits = result.logits
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                acc = flat_accuracy(logits , label_ids)
            valid_loss.append(loss.item())
            valid_accs.append(acc)
                
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        #打印验证集的loss和准确率
        print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        #如果验证集的准确率大于记录的最佳准确率，则更新最佳准确率，并保存模型参数
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(),PATH)
            best_acc =valid_acc
            stale = 0
        #如果太久没有进步则early stop
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break

    #测试部分
    print("Start testing")
    #首先测试Finetune前的准确率
    model =  BertForSequenceClassification.from_pretrained(bert_path, config=config)
    model.to(device)
    model.eval()
    prediction_accs = []
    #和验证部分类似
    for batch in tqdm(prediction_dataloader):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device) 
        with torch.no_grad():
            result = model(input_ids,
                            token_type_ids=None, 
                            attention_mask=input_mask,
                            return_dict=True)
            logits = result.logits
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            acc = flat_accuracy(logits , label_ids)
        prediction_accs.append(acc)
            
    prediction_acc = sum(prediction_accs) / len(prediction_accs)

    #打印Finetune前准确率
    print(f"Before finetune acc = {prediction_acc:.5f}")

    #载入保存好的模型参数，测试Finetune后的准确率
    model.load_state_dict(torch.load(PATH))
    prediction_accs = []
    for batch in tqdm(prediction_dataloader):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device) 
        with torch.no_grad():
            result = model(input_ids,
                            token_type_ids=None, 
                            attention_mask=input_mask,
                            return_dict=True)
            logits = result.logits
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            acc = flat_accuracy(logits , label_ids)
        prediction_accs.append(acc)
            
    prediction_acc = sum(prediction_accs) / len(prediction_accs)
    #打印Finetune后的准确率
    print(f"After finetune acc = {prediction_acc:.5f}")

#Main
if __name__ == '__main__':
    #可设置数据集类型
    name = "MRPC"
    #可设置bert类型
    bert_path = "roberta_base_localpath"
    #开始训练
    train(name,bert_path)