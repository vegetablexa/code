import LVJIANNAN____________dataloader as Data
import math
import numpy as np
from ordered_set import OrderedSet

import torch
from time import *
# typeList 是一个字典，键是关系，值是Trhead和Trtail
# relationList是关系的列表
# entityList是实体的列表
# entitytypeList是每个实体对应的类型的列表,没有进行区分
typeList,entitytypeList = Data.type_list,Data.entity2type


# 给出一个路径，打印出路径里面每种类型的权重，这个是2021年提出的，对不同路径的类型给出不同的权重
# def typeToWeight(typeroad):
#     types = typeroad.split('/')[1:]
#     weights = {}
#     fenmu = 0
#     for j in range(0, len(types)):
#         fenmu = fenmu + math.exp(j)
#
#     for i in range(0,len(types)):
#         # i代表的是当前的层数
#         # 同时types[i] 代表第i层的类型
#         weight = math.exp(i)/fenmu
#         weights[types[i]] = round(weight,2)
#     return weights
"""
def typeToWeight(typeroad):
    types = typeroad.split('/')[1:]
    weights = {}
    fenmu = 0
    for j in range(0, len(types)):
        fenmu = fenmu + math.exp(j)
    for i in range(0,len(types)):
        # i代表的是当前的层数
        # 同时types[i] 代表第i层的类型
        if(len(types)<3):
            if (i < ((len(types)) / 2)):
                # weight = (math.exp(i)-0.2) / fenmu
                weight = 0
            else:
                weight = ((math.exp(i)) / fenmu)+math.exp(0.5*i)
        else:
        # 不重要的类型权重直接定义为0
            if(i <=((len(types))/2)):
                weight = 0
            else:
                weight = ((math.exp(i))/fenmu) + math.exp(i)
        weights[types[i]] = round(weight,2)
    return weights
"""
# 给定一条路径 然后求出每个层次的权重是多少
def typeToWeight(typeroad):

    types = typeroad.split('/')[1:]
    weights = {}
    fenmu = 0
    for j in range(0, len(types)):
        fenmu = fenmu + math.exp(j)

    for i in range(0,len(types)):
        # i代表的是当前的层数
        # 同时types[i] 代表第i层的类型
        # 也就是说只有一层

        weight = (math.exp(i)+0.1)/fenmu
        weights[types[i]] = round(weight,2)
    return weights


# 对Trhead和TrTail按照阈值的公式进行处理  传出的是处理之后的Trhead，Trtail
def dealWithTypleList(typeListH,typeListT):

    minist = min(min(typeListH.values()),min(typeListT.values()))
    highest = max(max(typeListH.values()),max(typeListT.values()))
    standard = minist + 0.1 * (highest - minist)
    #print("阈值的标准是多少：{}".format(standard))
    for key in list(typeListH.keys()):
        if typeListH.get(key)<standard:
            e = typeListH.pop(key)
    for key in list(typeListT.keys()):
        if typeListT.get(key)<standard:
            p = typeListT.pop(key)
    # print("处理之前有多少种类型呢？")
    # print(len(typeListH))
    # print(len(typeListT))

    return typeListH,typeListT

# 传进来的是类型集和关系集合  出去的是处理过后的每一个关系对应的实体和类型的权重的字典
# 返回的是一个字典，字典的键是关系，值是一个列表，列表的每一位又是字典，字典的键是关系，值是基于层次的权重
def calculateWeight(typeList,relationList):
    dealRelationList = {}

    # 每一个关系都要清洗一下类型集合 每一个关系
    for i in range(0, len(relationList)):
        # 这里知识对Trhead进行了权重处理 typeTr是Trhead的路径集合 该关系下的
        list = [0,0]
        type_weight_dic0 = {}  # 头部类型集合
        type_weight_dic1 = {}  # 尾巴类型集合
        typeTr = typeList.get(relationList[i])[0] # Trhead
        typeTt = typeList.get(relationList[i])[1] # Trtail
        # 每个t就是Trhead一个路径了，返回一个字典，键是类型 值是该类型在该路径下的权重
        for t in typeTr:
            thisdic = typeToWeight(t)
            # cc是取出本次的路径的类型
            for cc in thisdic.keys():
                # 如果存在了
                if cc in type_weight_dic0.keys():
                    type_weight_dic0[cc] = min(thisdic.get(cc), type_weight_dic0.get(cc))
                else:
                    type_weight_dic0[cc] = thisdic.get(cc)

        # 对每个关系的尾市体的类型进行清洗
        for t in typeTt:
            thisdic = typeToWeight(t)
            # cc是取出本次的路径的类型
            for cc in thisdic.keys():
                # 如果存在了
                if cc in type_weight_dic1.keys():
                    type_weight_dic1[cc] = min(thisdic.get(cc), type_weight_dic1.get(cc))
                else:
                    type_weight_dic1[cc] = thisdic.get(cc)

        #list[0],list[1] = dealWithTypleList(type_weight_dic0, type_weight_dic1)
        list[0], list[1] = type_weight_dic0, type_weight_dic1
        # print("处理之前有多少种类型呢？")
        # print(len(list[0]))
        # print(len(list[1]))
        dealRelationList[relationList[i]] = list

    return dealRelationList

# 传进去的是去除噪声之后的关系的字典,还有关系的列表
# 返回值是当前还有的类型的列表
def create_type_list(del_relationlist,relationlist):
    type_list = ['0'] * 5000
    i = 0
    for rel in relationlist:
        list = del_relationlist.get(rel)
        # head = list[0].keys()
        # tail = list[1].keys()
        for type in list[0].keys():
            if(type in type_list):
                pass
            else:
                type_list[i] = type
                i = i + 1
        for type in list[1].keys():
            if(type in type_list):
                pass
            else:
                type_list[i] = type
                i = i + 1
    for i in range(0,len(type_list)):
        if(type_list[i] == '0'):
            type_list = type_list[0:i]
            break
    return type_list


def create_typeplus_list(etypelist,entityList,type_List):
    type_list = type_List
    #print(type_List[50])
    #print(type_List[2500])

    i = len(type_List)
    for e in entityList:
        list = etypelist.get(e)
        # head = list[0].keys()
        # tail = list[1].keys()
        #print(list)
        for type in list:
            if(type in type_list):
                pass
            else:
                type_list.append("0")
                type_list[i] = type
                i = i + 1

    for i in range(0,len(type_list)):
        if(type_list[i] == '0'):
            type_list = type_list[0:i]
            break
    return type_list


# 也就是核心矩阵的构建，relation * type 的核心矩阵
# 传入的是处理之后的类型的大字典,关系列表和类型列表
# 返回的是relation * type的矩阵 ，这里的值是基于层次的权重
def create_matrix(dealrelationlist,relationlist,type_plus_list):
    matrix = np.empty([len(relationList),len(type_plus_list)])
    matrix1 = np.empty([len(relationList),len(type_plus_list)])  # 头部类型权重
    #1345 * 4294
    # print("核心矩阵的维度是什么样的？")
    # print(matrix.shape)
    # 每一行

    for i in range(0,len(relationlist)):
        # 每一列
        Trhead = dealrelationlist.get(relationlist[i])[0]
        #print(Trhead)
        Trtail = dealrelationlist.get(relationlist[i])[1]
        for type in Trhead.keys():
            index = type_plus_list.index(type)
            #print(Trhead.get(type))
            matrix1[i][index] = Trhead.get(type)
        for type in Trtail.keys():
            index = type_plus_list.index(type)
            # print(Trhead.get(type))
            matrix[i][index] = Trtail.get(type)
            #matrix[rel][type] = round(Trtail.get(type), 2)
    return matrix,matrix1


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

# 构建实体个数和类型的矩阵 有的地方是1，没有该类型的地方是0
def create_entity_matrix(etypelist,type_list):
    # matrix = np.empty([len(relationList),len(type_list)])
    matrix1 = np.empty([len(entityList),len(type_list)])  # 实体包含的类型集
    #1345 * 4294
    # print("核心矩阵的维度是什么样的？")
    # print(matrix.shape)
    # 每一行

    for i in range(0,len(entityList)):
        # 每一列
        # Trhead = dealrelationlist.get(relationlist[i])[0]
        #print(Trhead)
        # 取到每个实体的类型集合
        Tentity = etypelist.get(entityList[i])
        for type in Tentity:
            if type not in type_List:
                continue
            else:
                index = type_List.index(type)
            #print(Trhead.get(type))
            matrix1[i][index] = 1
        # for type in Trtail.keys():
        #     index = type_plus_list.index(type)
        #     # print(Trhead.get(type))
        #     matrix[i][index] = Trtail.get(type)
        #     #matrix[rel][type] = round(Trtail.get(type), 2)
    return matrix1

# 传入的是两个类型集合就是针对一个关系和一个实体，传入类型的列表
# 计算两个集合的语义相似度 采用的是2017年的简单方法计算的语义相似度
def cal(Trelation,T):
    count0 = 0
    for t in Trelation:
        if (t in T):
            count0 = count0 + 1
        print(count0)

    return round(count0/len(Trelation),2)


# 传入的是两个类型集合就是针对一个关系和一个实体，传入类型的列表
# 计算两个集合的语义相似度 采用的是2021改进的相似度
def cal2021(Trelation,T,matrix,relation):
    weightsum1 = 0  # 计算分子
    weightsum2 = 0  # 计算分母
    for t in Trelation:
        weightsum2 = weightsum2 + matrix[relationList.index(relation)][type_List.index(t)]
        if (t in T):
            weightsum1 = weightsum1 + matrix[relationList.index(relation)][type_List.index(t)]
    # # print(round(weightsum1/weightsum2,2))
    # print(weightsum1)
    # print(weightsum2)
    # print("-----------------------------------------")
    return round(weightsum1/weightsum2,2)


# 传入的一次是类型列表 实体列表  处理之后的大字典  每个实体对应的类型的列表带有/的
# 计算相似度矩阵，计算了Trtail和type的矩阵，但是中间用的是2017年的简单计算办法
def calculate_simMatrix(relationlist,entity,typelist,entitytypeList,matrix,flag):
    #matrix = np.empty([len(relationlist), len(entity)])
    matrix1 = np.zeros([len(relationlist), len(entity)])
    #(1345, 14941) 矩阵的shape
    # 每一行，
    for i in range(0, len(relationlist)):
        # 对每一行进行操作
        for j in range(0,len(entity)):
            # 进入了每一
            # 取出该关系的Trhead和Trtail
            if(flag ==0):
            #Trhead = typelist.get(relationlist[i])[0]

            # Trtail的类型集和

                Trtail = typelist.get(relationlist[i])[1]
            else:
                Trtail = typelist.get(relationlist[i])[0]
            Trtail = list(Trtail.keys())
            #print(Trtail)

            Tentity = entitytypeList.get(entity[j])
            #print(Tentity)
            #print("------------------------------------")
            #matrix1[i][j] = cal(Trtail,Tentity)
            matrix1[i][j] = cal2021(Trtail,Tentity,matrix,relationlist[i])
    return matrix1




# 计算关系预测的
def calculate_simMatrix1(relationlist,entity,typelist,entitytypeList,matrix,flag):
    #matrix = np.empty([len(relationlist), len(entity)])
    matrix1 = np.zeros([len(entity),len(relationlist)])
    #(1345, 14941) 矩阵的shape
    # 每一行，0,len(entity[:10])
    for i in range(0, len(entity)):
        # 对每一行进行操作
        for j in range(0,len(relationlist)):
            # 进入了每一
            # 取出该关系的Trhead和Trtail
            if(flag ==0):
            #Trhead = typelist.get(relationlist[i])[0]

            # Trtail的类型集和

                Trtail = typelist.get(relationlist[j])[1]
            else:
                Trtail = typelist.get(relationlist[j])[0]
            Trtail = list(Trtail.keys())
            #print(Trtail)

            Tentity = entitytypeList.get(entity[i])
            #print(Tentity)
            #print("------------------------------------")
            #matrix1[i][j] = cal(Trtail,Tentity)
            matrix1[i][j] = cal2021(Trtail,Tentity,matrix,relationlist[j])
    return matrix1
# 传入的是带有/的实体对应的类型列表和实体列表
# 返回的是的是一个字典，字典的键是实体，值是实体的类型，分开的类型
def create_entity_type_dic(e2t,elist):
    entity_type_dic = {}
    for e in elist:
        type = set()
        if e2t.get(e):
            list = e2t.get(e)
        else:
            print("不在第一个文件里面")


        # if(not list is None):
        #     list = e2t2.get(e)
        for l in list:
            k = l.split("/")[1:]
            for ks in k:
                type.add(ks)
        entity_type_dic[e] = type
    return entity_type_dic


# 测试用代码，用哪个求那个
# 关系列表,类型集，处理后的TrheadTrtail,权重矩阵，flag为0测试尾部和关系
# 返回一个相似度
def ceshi(relationlist,typelist,entitytypeList,matrix,flag,en):
    lists = [0] * len(relationList)
    for i in range(0,len(relationList)):
        if (flag == 0):
            # Trhead = typelist.get(relationlist[i])[0]
            # Trtail的类型集和
            Trtail = typelist.get(relationlist[i])[1]
        else:
            Trtail = typelist.get(relationlist[i])[0]
        Trtail = list(Trtail.keys())
        # 实体的类型集
        Tentity = entitytypeList.get(entityList[en])
        lists[i] = cal2021(Trtail, Tentity, matrix, relationlist[i])

    return lists

def load_data():
    ent_set, rel_set = OrderedSet(), OrderedSet()
    for split in ['train', 'test', 'valid']:
        for line in open('./data/{}/{}.txt'.format("FB15k-237", split)):
            sub, rel, obj = map(str.lower, line.strip().split('\t'))
            ent_set.add(sub)
            rel_set.add(rel)
            ent_set.add(obj)



    # 给实体和关系进行编号,加上了反关系
    ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
    # print(self.ent2id)
    rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
    # print("先验模型关系编号")
    # print(rel2id)

    return ent2id,rel2id

entityList,relationList = load_data()
entityList = list(entityList)
relationList = list(relationList)
# print("新的类型集和")

# print(relationList)
# print(shitiliebiao)
# print(guanxiliebiao)
# print(len(shitiliebiao))
# print(len(guanxiliebiao))
# 处理过后的类型集和,也就是处理后的大字典，Trhead Trtail都包含了
deal_relationlist = calculateWeight(typeList,relationList)

# 237个关系
# print(deal_relationlist)
# print(len(deal_relationlist.keys()))
# 类型构成的集合 有1590个不同的类型
# type_List是关系类型有的类型集合
type_List = create_type_list(deal_relationlist,relationList)

# 处理过后只有53个类型了
# print(type_List)
# print(len(type_List))
# print(len(type_List))
# 每个实体对应的列表，这个列表存储的是分开的类型
etypelist = create_entity_type_dic(entitytypeList,entityList)

# type_plus_List是加上实体的类型集构成的最终的type集合
#type_plus_List = create_typeplus_list(etypelist,entityList,type_List)

# 1345 * 4294的矩阵
# 关系数目 * 类型数目的矩阵，此时里面的数据是基于层次分类的，核心矩阵
# 第一个是Trtail,第二个是Trhead的矩阵
#matrix,matrix1 = create_matrix(deal_relationlist,relationList,type_plus_List)

matrix,matrix_head = create_matrxi2(deal_relationlist,relationList,type_List)

matrix_entity = create_entity_matrix(etypelist,type_List)

# print(matrix)
# print(matrix.shape)

