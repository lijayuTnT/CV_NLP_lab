# 基于预训练模型的分类模型(bert-tiny,bert-base-uncased,roberta-base)

## WNLI（Winograd NLI）
Finetune基础参数：
```
batch_size = 16
optimizer = AdamW
scheduler = get_linear_schedule_with_warmup
learning_rate =  2e-5
eps = 1e-8
epochs = 2
```
|bert-tiny|1|2|3|4|5|average|
|----|----|----|----|----|----|----|
|Before Finetune|0.39732|0.54018|0.39732|0.48661|0.52232|0.46875|
|After Finetune|0.57142|0.61310|0.46577|0.48214| 0.58631|0.543748|

|bert-base-uncased|1|2|3|4|5|average|
|----|----|----|----|----|----|----|
|Before Finetune|0.51786|0.47173|0.48214|0.43006|0.50893|0.482144|
|After Finetune|0.56548|0.60268|0.45536|0.43006|0.54018|0.518752|

|Roberta-base|1|2|3|4|5|average|
|----|----|----|----|----|----|----|
|Before Finetune|0.39732| 0.47173|0.63988|0.56548|0.43452|0.501786|
|After Finetune|0.43452| 0.56548|0.67884| 0.60268|0.48214|0.552732|

||Before Finetune   |After Finetune|
|----|----|----|
|bert-tiny|0.46875|0.543748|
|bert-base-uncased|0.482144|0.518752|
|Roberta-base|0.501786|0.552732|


## MRPC(Microsoft Research Paraphrase Corpus)

Finetune参数：
```
batch_size = 32
optimizer = AdamW
scheduler = get_linear_schedule_with_warmup
learning_rate =  2e-5
eps = 1e-8
epochs = 2
```
|bert-tiny|1|2|3|4|5|average|
|----|----|----|----|----|----|----|
|Before Finetune|0.32193|0.66513|0.48262|0.66050|0.38406|0.502848|
|After Finetune|0.68066|0.67012|0.68066|0.66661|0.68066|0.675742|

|bert-base-uncased|1|2|3|4|5|average|
|----|----|----|----|----|----|----|
|Before Finetune|0.67844|0.67474|0.60991|0.67604| 0.54401|0.636628|
|After Finetune|0.78994|0.82119|0.83672|0.80677|0.80677|0.812278|

|Roberta-base|1|2|3|4|5|average|
|----|----|----|----|----|----|----|
|Before Finetune|0.67714|0.67363|0.32988|0.31583|0.31232|0.46176|
|After Finetune|0.68066| 0.68066|0.67714|0.68417|0.68066|0.680658|

||Before Finetune   |After Finetune|
|----|----|----|
|bert-tiny|0.502848|0.675742|
|bert-base-uncased|0.636628|0.812278|
|Roberta-base|0.46176|0.680658|

## Conclusion
Accuracy： bert-base-uncased > Roberta-base > bert-tiny

Time:Roberta-base >bert-base-uncased>bert-tiny

Size : Roberta-base >bert-base-uncased>bert-tiny