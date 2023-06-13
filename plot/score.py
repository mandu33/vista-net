


from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


true_lable = [0, 4, 0, 0, 5, 2, 1, 2, 0, 3, 2, 3, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 3, 1, 0, 0, 4]
prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


measure_result = classification_report(true_lable, prediction)
print('measure_result = \n', measure_result)

print("----------------------------- precision（精确率）-----------------------------")
precision_score_average_None = precision_score(true_lable, prediction, average=None)
precision_score_average_micro = precision_score(true_lable, prediction, average='micro')
precision_score_average_macro = precision_score(true_lable, prediction, average='macro')
precision_score_average_weighted = precision_score(true_lable, prediction, average='weighted')
print('precision_score_average_None = ', precision_score_average_None)
print('precision_score_average_micro = ', precision_score_average_micro)
print('precision_score_average_macro = ', precision_score_average_macro)
print('precision_score_average_weighted = ', precision_score_average_weighted)

print("\n\n----------------------------- recall（召回率）-----------------------------")
recall_score_average_None = recall_score(true_lable, prediction, average=None)
recall_score_average_micro = recall_score(true_lable, prediction, average='micro')
recall_score_average_macro = recall_score(true_lable, prediction, average='macro')
recall_score_average_weighted = recall_score(true_lable, prediction, average='weighted')
print('recall_score_average_None = ', recall_score_average_None)
print('recall_score_average_micro = ', recall_score_average_micro)
print('recall_score_average_macro = ', recall_score_average_macro)
print('recall_score_average_weighted = ', recall_score_average_weighted)

print("\n\n----------------------------- F1-value-----------------------------")
f1_score_average_None = f1_score(true_lable, prediction, average=None)
f1_score_average_micro = f1_score(true_lable, prediction, average='micro')
f1_score_average_macro = f1_score(true_lable, prediction, average='macro')
f1_score_average_weighted = f1_score(true_lable, prediction, average='weighted')
print('f1_score_average_None = ', f1_score_average_None)
print('f1_score_average_micro = ', f1_score_average_micro)
print('f1_score_average_macro = ', f1_score_average_macro)
print('f1_score_average_weighted = ', f1_score_average_weighted)

