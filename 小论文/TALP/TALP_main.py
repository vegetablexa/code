import os
import time
import argparse
import numpy as np
import LVJIANNAN____________dataloader as Data
import torch
import LVJIANNAN___model as M
import LVJIANNAN_________prior as P
from helper import *
import torch.nn

import torch.nn.functional as F
import torch.optim as optim

# 加载参数
parser = argparse.ArgumentParser(description="Parser For Arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--name",            			default='testrun_'+str(uuid.uuid4())[:8],	help='Name of the experiment')
parser.add_argument("--l2", type=float, default=0.0, help='L2 regularization')
# --lr     default=0.001
parser.add_argument("--lr", type=float, default=0.001, help='Learning Rate')
parser.add_argument('--restore', dest="restore", action='store_true',
                    help='Restore from the previously saved model')
parser.add_argument("--epoch", dest='max_epochs', default=10, type=int, help='Maximum number of epochs')
parser.add_argument("--pre_lr", type=float, default=0.01, help='pre_Learning Rate')
# Logging parameters
parser.add_argument('--logdir',    	dest="log_dir",       	default='./log/',               		help='Log directory')
parser.add_argument('--config',    	dest="config_dir",    	default='./config/',            		help='Config directory')
args = parser.parse_args()

# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 加载使用日志文件
logger = get_logger(args.name, args.log_dir, args.config_dir)
logger.info(vars(args))




# Model and optimizer
# matrix是尾的权重矩阵  matrix1是头的权重矩阵
matrix = torch.Tensor(P.matrix).to(device)
#matrix1 = torch.Tensor(P.matrix1).to(device)
# 传入的是头的权重矩阵和 尾的权重矩阵
# 实体类型的01矩阵 14951*1590
matrix_entity_01 =torch.Tensor(P.matrix_entity).to(device)
model = M.model(matrix,matrix_entity_01).to(device)




# 测试用的小代码

'''
dataloader = Data.test_loader
for data in dataloader:
    pass

start_time = time.time()
print("*********************************************")
print("测试一下模型哦")
head = data[0]
rel = data[1]
tail = data[2]

# print("本次预测的三元组是：")
# print(head)
# print(rel)
# print(tail)
# 存储的是训练数据的真实标签
label = []
# 存储了真实标签，同时构建测试batch的128个关系对应的类型矩阵
test_matrix = []
for i in range(0,len(rel)):
    label.append(P.entityList.index(tail[i]))
    # 取出每个关系的角标
    index = P.relationList.index(rel[i])
    test_matrix.append(P.matrix[index])

test_matrix = np.matrix(test_matrix)


#matrix_entity1 = np.zeros([len(head),len(P.type_List)],dtype=float)
#matrix_entity2 = np.zeros([len(head),len(P.type_List)],dtype=float)

# trhead = np.zeros([len(head),len(P.type_plus_List)],dtype=float)
# trtail = np.zeros([len(head),len(P.type_plus_List)],dtype=float)

# 构建好了头和尾部的参数矩阵
# for i in range(0,len(head)):
#     # type返回的是实体的类型集合
#     type = P.etypelist.get(head[i])
#     # print("打印了头实体 还有它的类型集和个数")
#     # print(head)
#     # print(type)
#     # print(len(type))
#     if(type == None):
#         matrix_entity1[i][:] = 0
#     else:
#         for j in type:
#             matrix_entity1[i][P.type_plus_List.index(j)] = 1

# for i in range(0,len(tail)):
#     type = P.etypelist.get(tail[i])
#     if (type == None):
#         matrix_entity2[i][:] = 0
#     else:
#         for j in type:
#             if j not in P.type_List:
#                 continue
#             else:
#                 matrix_entity2[i][P.type_List.index(j)] = 1


# 构建Trhead 和 Trtail的矩阵呢
# for i in range(0,len(rel)):
#     type = P.deal_relationlist.get(rel[i])[0]
#     type = list(type.keys())
#     if (type == None):
#         trhead[i][:] = 0
#     else:
#         for j in type:
#             trhead[i][P.type_plus_List.index(j)] = 1
#
# for i in range(0,len(rel)):
#     type = P.deal_relationlist.get(rel[i])[1]
#     type = list(type.keys())
#     if (type == None):
#         trtail[i][:] = 0
#     else:
#         for j in type:
#             trtail[i][P.type_plus_List.index(j)] = 1

# matrix_entity1 = torch.tensor(matrix_entity1)
# matrix_entity1 = matrix_entity1.to(torch.float32)
# matrix_entity2 = torch.tensor(matrix_entity2)
# matrix_entity2 = matrix_entity2.to(torch.float32)

# trhead = torch.tensor(trhead)
# trhead = trhead.to(torch.float32)
# trtail = torch.tensor(trtail)
# trtail = trtail.to(torch.float32)

# 预测除的矩阵
#pred = model.forward(head,rel,tail)

# 传进来的是 头实体表示矩阵,尾巴实体的表示矩阵，
test_matrix = torch.tensor(test_matrix)
test_matrix = test_matrix.to(torch.float32)
test_matrix.to(device)
fenmu_matrix = np.empty([len(head),14941])
y = test_matrix.sum(1)
for i in range(len(y)):
    fenmu_matrix[i][:] = y[i]

# print(fenmu_matrix)
# print(fenmu_matrix.shape)

fenmu_matrix = torch.tensor(fenmu_matrix)
fenmu_matrix = fenmu_matrix.to(torch.float32)
fenmu_matrix = fenmu_matrix.to(device)

pred = model.forward(fenmu_matrix,test_matrix)
end_time = time.time()
print("每个batch_size计算要多久{}".format(end_time-start_time))


# print("预测的结果是:")
# print(pred)
# print("预测出来的最大值是？")
# print(pred.argmax(1))
# print("预测的最大值是:{}".format(torch.argmax(pred)))
# print("这个三元组的真实值是:{}".format(label))
# ic = 0
# with torch.no_grad():
#     model.eval()
#     # 存储预测出的最大值
#     for i in range(0,len(head)):
#         # print(head[i])
#         # print(rel[i])
#         # print(tail[i])
#         key = max(pred[i])
#         if(key == pred[i][label[i]]):
#             print("命中了~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#             ic = ic + 1
#
#         # Lists = []
#         # key = max(pred[i])
#         # for j in range(0,len(pred[i])):
#         #     if pred[i][j] == key:
#         #         Lists.append(j)
#         # print(Lists)
#         # print(label[i])
#
#
#         # print("差了多少呢？？？？？？？？？？？？？？？？？？？？？")
#         #
#         # print(key - pred[i][(label[i])])
#         # if(label[i] in Lists):
#         #     print("++++++++++++++++++++++++++++命中++++++++++++++++++")
#         #     ic = ic+1
#
# # # 计算loss
# print("{}命中了{}".format(len(head),ic))
# #print(P.type_plus_List[16])
# print("********************************************")
# end_time = time.time()
# print("32个数据连测试用了多久{}".format(end_time-start_time))
'''
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=args.pre_lr, weight_decay=args.l2)

# 加载数据
dataloader = Data.test_loader

valid_dataloader = Data.valid_loader

test_dataloader = Data.test_loader

# 保存参数的函数 #############
def save_model(save_path,best_mingzhong):
    """
    Function to save a model. It saves the model parameters, best validation scores,
    best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

    Parameters
    ----------
    save_path: path where the model is saved

    Returns
    -------
    """
    state = {
        'state_dict': model.state_dict(),
        'best_mingzhong': best_mingzhong,
        'optimizer': optimizer.state_dict(),
        'args'	: vars(args)
    }
    torch.save(state, save_path)

# 加载模型  ############
def load_model(self, load_path):
    """
    Function to load a saved model

    Parameters
    ----------
    load_path: path to the saved model

    Returns
    -------
    """
    state = torch.load(load_path)
    state_dict = state['state_dict']
    best_val_mrr = state['best_val']['mrr']
    best_val = state['best_val']

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(state['optimizer'])

# 评估先验模型 #############
# 算评价指标的地方

def evaluate_priori_probability(self, split, epoch=0):
    valid_dataset = Data.valid_loader
    left_results = self.predict_priori_probability(split=split, mode='tail_batch')
    results = get_combined_results(left_results)
    logger.info(
        '[Epoch {} {}]: pre_MRR: Tail : {:.5}'.format(epoch, split, results['mrr'],
                                                      ))
    return results



# 训练
def run_model(data,flag,epoch):
    if (flag == 1):

        model.train()
        optimizer.zero_grad()
        # 用于存储loss的
        # 每次都吧epoch的loss加进去了
        losses = []
        mingzhonglv = [0]
        number = 0
        train_iter = iter(data)
        # 每次读取出一个batch 128
        for step, batch in enumerate(train_iter):
            head = batch[0]
            rel = batch[1]
            tail = batch[2]
            # 每一个batch的标签  也就是真实值
            relindex = []
            label = []
            label_matrix = np.zeros([len(head),len(P.entityList)])
            # 处理训练集中不在里面的
            # for i in range(0, len(rel)):
            #     if tail[i] not in P.entityList:
            #         pass
            #     else:
            #         label.append(P.entityList.index(tail[i]))
            #         label_matrix[i][P.entityList.index(tail[i])] = 1
            # 前向传播 这个是每个batch实体对应的类型矩阵的构建
            # 顺便构建真实标签的矩阵算loss
            test_matrix = []
            for i in range(0, len(rel)):
                relindex.append(P.relationList.index(rel[i]))
                ################################################## 对缺失数据的处理是有问题的
                if tail[i] not in P.entityList:
                    label.append(0)
                else:
                    label.append(P.entityList.index(tail[i]))
                    label_matrix[i][P.entityList.index(tail[i])] = 1
                # 取出每个关系的角标
                index = P.relationList.index(rel[i])
                test_matrix.append(P.matrix[index])

            # 构建好了labelmatrix
            # print(label_matrix)
            # print(label_matrix.sum(1))
            label_matrix = torch.tensor(label_matrix,dtype = torch.float32)
            label_matrix = label_matrix.to(device)


            test_matrix = np.matrix(test_matrix)
            test_matrix = torch.tensor(test_matrix)
            test_matrix = test_matrix.to(torch.float32)
            test_matrix.to(device)

            fenmu_matrix = np.empty([len(head), 14941])
            y = test_matrix.sum(1)
            for i in range(len(y)):
                fenmu_matrix[i][:] = y[i]

            # print(fenmu_matrix)
            # print(fenmu_matrix.shape)

            fenmu_matrix = torch.tensor(fenmu_matrix)
            fenmu_matrix = fenmu_matrix.to(torch.float32)
            fenmu_matrix = fenmu_matrix.to(device)



            pred = model.forward(fenmu_matrix, relindex)
            pres = pred
            #pred = model.forward(matrix_entity1, matrix_entity2)

            # 存储有可能的值
            # predlist = []
            # for i in range(0,pred.shape[0]):
            #     pmax = torch.max(pred[i])
            #     jieguo = torch.where(pred[i]== pmax,1,0)
            #     b = (torch.nonzero(jieguo, as_tuple=False).cpu()).numpy()
            #     b = b.ravel()
            #     predlist.append(b)

            # 这个就是每次预测出来的标签  label是真实标签
            predictlabel = torch.argmax(pred,1)

            #pred = pred.tensor(pred,dtype = float)
            # 穿进去的是 预测标签 真实标签  和预测矩阵
            #pred = pred.to(torch.float32)
            pred = torch.sigmoid(pred)
            # print(pred.shape)
            # print(label_matrix.shape)
            loss,mingzhong = model.loss(pred,label_matrix,label,predictlabel)
            #loss.astype(torch.tensor)
            loss.to(device)
            loss.backward()
            optimizer.step()
            # if (mingzhong > max(mingzhonglv)):
            #     pass
            #
            #     torch.save(model, "model{}.pkl".format(number))
            #     print(max(mingzhonglv))
            #     print("保存好了一个模型")
            #     number = number + 1
            # print("在外面哦---------------------------")
            # print(mingzhong)
            # print(max(mingzhonglv))

            losses.append(loss.item())
            mingzhonglv.append(mingzhong)

            if step % 200 == 0:
                torch.save(model, "model{}.pkl".format(number))
                number = number + 1
                print("每1000个step保存一次模型 保存模型成功了")

            if step % 50 == 0:
                print(
                    '[Epoch{} : 训练 {}]: Train Loss:{:.5} 命中率是{}'.format(epoch,step, np.mean(losses),mingzhong
                                                          ))


                # torch.save(model, "model{}.pkl".format(number))
                # print(max(mingzhonglv))
                #print("保存好了一个模型")
                # number = number + 1
        loss11 = np.mean(losses)
        hit = np.mean(mingzhonglv)
        print('[0]:  Training Loss:{:.4},命中率:{} \n'.format(loss11,hit))

    else:
        print("进入了最终测试模块咯~~~~~~~~~~~~~~~~~~~~~~~~")
        with torch.no_grad():
            model.eval()
            optimizer.zero_grad()
            # 用于存储loss的
            # 每次都吧epoch的loss加进去了
            losses = []
            mingzhonglv = [0]
            number = 0
            train_iter = iter(data)
            # 每次读取出一个batch 128
            for step, batch in enumerate(train_iter):
                head = batch[0]
                rel = batch[1]
                tail = batch[2]
                # 每一个batch的标签  也就是真实值
                relindex = []
                label = []
                label_matrix = np.zeros([len(head), len(P.entityList)])
                # 处理训练集中不在里面的
                # for i in range(0, len(rel)):
                #     if tail[i] not in P.entityList:
                #         pass
                #     else:
                #         label.append(P.entityList.index(tail[i]))
                #         label_matrix[i][P.entityList.index(tail[i])] = 1
                # 前向传播 这个是每个batch实体对应的类型矩阵的构建
                # 顺便构建真实标签的矩阵算loss
                test_matrix = []
                for i in range(0, len(rel)):
                    relindex.append(P.relationList.index(rel[i]))
                    ################################################## 对缺失数据的处理是有问题的
                    if tail[i] not in P.entityList:
                        label.append(0)
                    else:
                        label.append(P.entityList.index(tail[i]))
                        label_matrix[i][P.entityList.index(tail[i])] = 1
                    # 取出每个关系的角标
                    index = P.relationList.index(rel[i])
                    test_matrix.append(P.matrix[index])

                # 构建好了labelmatrix
                # print(label_matrix)
                # print(label_matrix.sum(1))
                label_matrix = torch.tensor(label_matrix, dtype=torch.float32)
                label_matrix = label_matrix.to(device)

                test_matrix = np.matrix(test_matrix)
                test_matrix = torch.tensor(test_matrix)
                test_matrix = test_matrix.to(torch.float32)
                test_matrix.to(device)

                fenmu_matrix = np.empty([len(head), 14941])
                y = test_matrix.sum(1)
                for i in range(len(y)):
                    fenmu_matrix[i][:] = y[i]

                # print(fenmu_matrix)
                # print(fenmu_matrix.shape)

                fenmu_matrix = torch.tensor(fenmu_matrix)
                fenmu_matrix = fenmu_matrix.to(torch.float32)
                fenmu_matrix = fenmu_matrix.to(device)

                pred = model.forward(fenmu_matrix, relindex)
                pres = pred
                # pred = model.forward(matrix_entity1, matrix_entity2)

                # 存储有可能的值
                # predlist = []
                # for i in range(0,pred.shape[0]):
                #     pmax = torch.max(pred[i])
                #     jieguo = torch.where(pred[i]== pmax,1,0)
                #     b = (torch.nonzero(jieguo, as_tuple=False).cpu()).numpy()
                #     b = b.ravel()
                #     predlist.append(b)

                # 这个就是每次预测出来的标签  label是真实标签
                predictlabel = torch.argmax(pred, 1)

                # pred = pred.tensor(pred,dtype = float)
                # 穿进去的是 预测标签 真实标签  和预测矩阵
                # pred = pred.to(torch.float32)
                pred = torch.sigmoid(pred)
                # print(pred.shape)
                # print(label_matrix.shape)
                loss, mingzhong = model.loss(pred, label_matrix, label, predictlabel)
                # loss.astype(torch.tensor)
                loss.to(device)

                # loss.backward()
                # optimizer.step()

                # if (mingzhong > max(mingzhonglv)):
                #     pass
                #
                #     torch.save(model, "model{}.pkl".format(number))
                #     print(max(mingzhonglv))
                #     print("保存好了一个模型")
                #     number = number + 1
                # print("在外面哦---------------------------")
                # print(mingzhong)
                # print(max(mingzhonglv))

                losses.append(loss.item())
                mingzhonglv.append(mingzhong)

                if step % 100 == 0:
                    print(
                        '[训练 {}]: Train Loss:{:.5} 命中率是{}'.format(step, np.mean(losses), mingzhong
                                                                  ))
                    # torch.save(model, "model{}.pkl".format(number))
                    # print(max(mingzhonglv))
                    # print("保存好了一个模型")
                    # number = number + 1
            loss11 = np.mean(losses)
            hit = np.mean(mingzhonglv)
            print('[0]:  Training Loss:{:.4},命中率:{} \n'.format(loss11, hit))

    return loss

# 就是验证调整加上找到最佳的模型来进行最后的测试
def fit(args):
    best_val_mrr, best_val, best_epoch = 0., {}, 0.
    val_mrr = 0
    save_path = os.path.join('./torch_saved', args.name)
    pre_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=1,
                                                               factor=0.5,
                                                               patience=1, min_lr=5e-05)

    # 是否要保存模型
    if args.restore:
        # load_model(save_path)
        logger.info('Successfully Loaded previous model')

    start_train = time.time()
#     for epoch in range(0,10):
#         train_loss = run_model(dataloader,1,epoch)
#         pre_scheduler.step(train_loss)
# #        val_results = evaluate_priori_probability('valid', epoch)
#     print('[当前Epoch的时间 {}]:', format(time.time() - start_train))



    # # 开始测试  传入flag标识是验证集合
    print("开始验证")
    # for i in range(0,100):
    #     if(i%3==0):
    torch.load("model2.pkl")
    print("加载模型{}成功".format(0))
    best_mingzhong = run_model(test_dataloader,0,0)
    print("-------------------------------------------------------------")
    print('[当前Epoch的时间 {}]:', format(time.time() - start_train))
"""
    if val_results['mrr'] > self.best_val_mrr:
        self.best_val = val_results
        self.best_val_mrr = val_results['mrr']
        self.best_epoch = epoch
        self.save_model(save_path)
    print('[当前Epoch的时间 {}]:', format(time.time() - start_train))
    print('[Epoch {}]:  Training Loss: {:.5},  best MRR: {:.5}, \n\n\n'.format(epoch, train_loss,
                                                                                          self.best_val_mrr))

    print('Loading best model, evaluating on test data')
    #load_model(save_path)
"""


fit(args)