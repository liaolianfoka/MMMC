import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from datetime import timedelta
import os
learning_rate = 1e-4  # 学习率
num_epochs = 300  # epoch数
require_improvement = 500  # 若超过2000batch效果还没提升，则提前结束训练
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, train_iter, val_iter, test_iter, device, args, loss_func):
    start_time = time.time()
    model.train()
    num = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')  # 跟踪验证集最小的loss 或 最大f1-score或准确率
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(num_epochs):
        with open(args.save_file, "a+") as f:
            f.write('\nEpoch [{}/{}]'.format(epoch + 1, num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            Eimgemb = trains[0] # 32*20
            Eimgvit = trains[1]
            Simgemb = trains[2]
            Simgvit = trains[3]
            Timgemb = trains[4]
            Timgvit = trains[5]
            textfeature = trains[6]
            sourcefeature = trains[7]
            targetfeature = trains[8]
            outputs = model(Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature)
            model.zero_grad()    # 清空梯度
            loss = loss_func(outputs, labels.long())  # 计算交叉熵损失
            #l2_lambda = 0.001
            #l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            #loss += l2_lambda * l2_norm
            loss.backward()  # 计算梯度
            optimizer.step() # 更新参数
            if total_batch % 10 == 0:  # 每100个batch 计算一下在验证集上的指标 或者像之前项目中那样 每一个epoch在验证集上计算相关指标
                _, predic = outputs.cpu().max(1)
                train_acc = metrics.accuracy_score(labels.cpu(), predic)  # 当前batch上训练集的准确率 因为是类别均衡数据集，所以可以直接用准确率作评估指标 不然使用 macro f1-score
                dev_acc, dev_loss = evaluate2(device, args, model, val_iter, test_iter, dev_best_loss, loss_func)#计算此时模型在验证集上的损失和准确率
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    last_improve = total_batch
                    torch.save(model, args.save_file+'_model_'+str(num)+'.pth')
                    if os.path.exists(args.save_file+'_model_'+str(num-3)+'.pth'):
                        os.remove(args.save_file+'_model_'+str(num-3)+'.pth')
                    num += 1
                    with open(args.save_file, "a+") as f:
                        f.write("****imporve****\n")
                #if dev_acc > 0.31:
                #    dev_best_loss = dev_loss  # 更新验证集最小损失
                #    improve = '*'  # 效果提升标志
                #    last_improve = total_batch  # 计算上次提升 位于哪个batch
                #    test(device, args, model, val_iter, val=True)
                #    test(device, args, model, test_iter, val=False)
                #else:
                #    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}'
                with open(args.save_file, "a+") as f:
                    f.write(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc)+"\n")
                model.train()  # 回到训练模式
            total_batch += 1
            if total_batch - last_improve > require_improvement:  # 如果长期没有提高 就提前终止
                return
                # 验证集loss超过2000batch没下降，结束训练
            #    print("No optimization for a long time, auto-stopping...")
            #    flag = True
            #    break
        #if flag:
        #    break
    # test(model, val_iter)
    #test(device, args, model, val_iter, val=True)
    #test(device, args, model, test_iter, val=False)  # 模型训练结束后 进行测试


def evaluate2(device, args, model, val_iter, test_iter, dev_best_loss, loss_func):
    model.eval()
    val_loss_total = 0
    val_predict_all = np.array([], dtype=int)
    val_labels_all = np.array([], dtype=int)

    test_loss_total = 0
    test_predict_all = np.array([],dtype=int)
    test_labels_all = np.array([], dtype=int)

    with torch.no_grad():  
        for (tests, labels) in val_iter:
            Eimgemb = tests[0] # 32*20
            Eimgvit = tests[1]
            Simgemb = tests[2]
            Simgvit = tests[3]
            Timgemb = tests[4]
            Timgvit = tests[5]
            textfeature = tests[6]
            sourcefeature = tests[7]
            targetfeature = tests[8]
            outputs = model(Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature)
            loss = loss_func(outputs, labels.long())
            val_loss_total += loss
            _, predic = outputs.cpu().max(1)
            val_labels_all = np.append(val_labels_all, labels.cpu())
            val_predict_all = np.append(val_predict_all, predic)
        val_acc = metrics.accuracy_score(val_labels_all, val_predict_all)
        
        if (val_loss_total / len(val_iter)) > dev_best_loss:
            return val_acc, val_loss_total / len(val_iter)
        
        for (tests, labels) in test_iter:
            Eimgemb = tests[0] # 32*20
            Eimgvit = tests[1]
            Simgemb = tests[2]
            Simgvit = tests[3]
            Timgemb = tests[4]
            Timgvit = tests[5]
            textfeature = tests[6]
            sourcefeature = tests[7]
            targetfeature = tests[8]
            outputs = model(Eimgemb, Eimgvit, Simgemb, Simgvit, Timgemb, Timgvit, textfeature, sourcefeature, targetfeature)
            loss = loss_func(outputs, labels.long())
            test_loss_total += loss
            _, predic = outputs.cpu().max(1)
            test_labels_all = np.append(test_labels_all, labels.cpu())
            test_predict_all = np.append(test_predict_all, predic)
            test_acc = metrics.accuracy_score(test_labels_all, test_predict_all)
        #if test_acc < args.test_acc:
        #return val_acc, val_loss_total / len(val_iter)
    
    if args.label == 2:
        target_names = [str(i) for i in range(1, 8)]
    elif args.label == 3:
        target_names = [str(i) for i in range(1, 6)]
    elif args.label == 4:
        target_names = [str(i) for i in range(0, 4)]
        
    val_report = metrics.classification_report(val_labels_all, val_predict_all, target_names=target_names, digits=4)
    val_confusion = metrics.confusion_matrix(val_labels_all, val_predict_all)
    test_report = metrics.classification_report(test_labels_all, test_predict_all, target_names=target_names, digits=4)
    test_confusion = metrics.confusion_matrix(test_labels_all, test_predict_all)

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}\n'
    with open(args.save_file, 'a+', encoding='utf-8') as f:
        f.write("\nval\n")
        f.write(msg.format(val_loss_total / len(val_iter), val_acc))
        f.write("Precision, Recall and F1-Score...\n")
        f.write(val_report)
        f.write("\nConfusion Matrix...\n")
        f.write(str(val_confusion))
        f.write("\ntest\n")
        f.write(msg.format(test_loss_total / len(test_iter), test_acc))
        f.write("Precision, Recall and F1-Score...\n")
        f.write(test_report)
        f.write("\nConfusion Matrix...\n")
        f.write(str(test_confusion))
    return val_acc, val_loss_total / len(val_iter)