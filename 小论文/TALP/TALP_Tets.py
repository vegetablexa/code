# 先验模型的部分 存储main函数的部分
import torch
import numpy as np
import LVJIANNAN_________prior as P
import LVJIANNAN___model as M




#  构建一个头尾融合的矩阵  尾部关系在上 头部在下面

def create_matrxi2(dealrelationlist,relationlist,type_list):
    matrix = np.empty([len(relationList),len(type_list)])
    matrix1 = np.empty([len(relationList),len(type_list)])  # 头部类型权重
    #1345 * 1590
    #print(matrix.shape)
    # 每一行

    for i in range(0,len(relationlist)):
        # 每一列
        Trhead = dealrelationlist.get(relationlist[i])[0]
        #print(Trhead)
        Trtail = dealrelationlist.get(relationlist[i])[1]
        for type in Trhead.keys():
            index = type_list.index(type)
            #print(Trhead.get(type))
            matrix1[i][index] = Trhead.get(type)
        for type in Trtail.keys():
            index = type_list.index(type)
            # print(Trhead.get(type))
            matrix[i][index] = Trtail.get(type)
            #matrix[rel][type] = round(Trtail.get(type), 2)
    #print(matrix)
    #print(matrix1)
    return matrix,matrix1





# 训练时用的头尾融合
def prior_calcute_tail(sub, rel, tail):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # ##？？？？ 按照新的方法先对实体和关系编号，然后构建新的实体01矩阵和关键矩阵
    matrix = torch.Tensor(P.matrix).to(device)

    # 传入的是头的权重矩阵和 尾的权重矩阵
    # 实体类型的01矩阵 14951*1590
    matrix_entity_01 = torch.Tensor(P.matrix_entity).to(device)
    model = M.model(matrix, matrix_entity_01).to(device)

    # 注意一下参数是否正确

    # losses = []
    # mingzhonglv = [0]
    # 进来一批数据
    # 一批头尾巴还有关系
    head = sub
    rel = rel
    tail = tail
    # 每一个batch的标签  也就是真实值
    relindex = []
    # label = []
    # label_matrix = np.zeros([len(head), len(P.entityList)])

    test_matrix = []
    for i in range(0, len(rel)):
        # 传进来的关系是按照interacte中的关系来定义的，

        relindex.append(rel[i].item())

        # if tail[i] not in P.entityList:
        # 	label.append(0)
        # else:
        # 	label.append(P.entityList.index(tail[i]))
        # 	label_matrix[i][P.entityList.index(tail[i])] = 1
        # 取出每个关系的角标
        index = rel[i]
        test_matrix.append(P.matrix[index])

    # 构建好了labelmatrix

    # label_matrix = torch.tensor(label_matrix, dtype=torch.float32)
    # label_matrix = label_matrix.to(device)

    test_matrix = np.matrix(test_matrix)
    test_matrix = torch.tensor(test_matrix)
    test_matrix = test_matrix.to(torch.float32)
    test_matrix.to(device)

    fenmu_matrix = np.empty([len(head), 14541])
    y = test_matrix.sum(1)
    for i in range(len(y)):
        fenmu_matrix[i][:] = y[i]

    fenmu_matrix = torch.tensor(fenmu_matrix)
    fenmu_matrix = fenmu_matrix.to(torch.float32)
    fenmu_matrix = fenmu_matrix.to(device)

    pred = model.forward(fenmu_matrix, relindex)

    return pred


# 根据尾部实体来预测头
def prior_calcute_head(self, sub, rel, tail):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # ##？？？？ 按照新的方法先对实体和关系编号，然后构建新的实体01矩阵和关键矩阵
    matrix_head = torch.Tensor(P.matrix_head).to(device)

    # 传入的是头的权重矩阵和 尾的权重矩阵
    # 实体类型的01矩阵 14951*1590
    matrix_entity_01 = torch.Tensor(P.matrix_entity).to(device)
    model = M.model(matrix_head, matrix_entity_01).to(device)

    # 注意一下参数是否正确
    # 进来一批数据
    # 一批头尾巴还有关系
    head = sub
    rel = rel
    tail = tail
    # 每一个batch的标签  也就是真实值
    relindex = []
    test_matrix = []
    for i in range(0, len(rel)):
        # 传进来的关系是按照interacte中的关系来定义的，

        relindex.append(rel[i].item())

        # if head[i] not in P.entityList:
        # 	label.append(0)
        # else:
        # 	label.append(P.entityList.index(head[i]))
        # 	# label_matrix[i][P.entityList.index(head[i])] = 1
        # 取出每个关系的角标
        index = rel[i]
        test_matrix.append(P.matrix_head[index])

    # 构建好了labelmatrix

    # label_matrix = torch.tensor(label_matrix, dtype=torch.float32)
    # label_matrix = label_matrix.to(device)

    test_matrix = np.matrix(test_matrix)
    test_matrix = torch.tensor(test_matrix)
    test_matrix = test_matrix.to(torch.float32)
    test_matrix.to(device)

    fenmu_matrix = np.empty([len(head), 14541])
    y = test_matrix.sum(1)
    for i in range(len(y)):
        fenmu_matrix[i][:] = y[i]

    fenmu_matrix = torch.tensor(fenmu_matrix)
    fenmu_matrix = fenmu_matrix.to(torch.float32)
    fenmu_matrix = fenmu_matrix.to(device)

    pred = model.forward(fenmu_matrix, relindex)

    return pred
