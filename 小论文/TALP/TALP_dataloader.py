import torchvision
from torch.utils.data import DataLoader

# 用来解析训练集的每一行的数据
def parse_line(line):          # 上来进来的是 FB15K数据集的train文件，解析一行，返回头尾实体和关系
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(),line[1].strip(), line[2].strip()
    return e1,relation ,e2

# 用来加载训练集文件的，返回的是一个列表，存储了三元组数据
def load_data(filename):
    triples_data = []  # 存储三元组数据
    unique_entities = set()  # 存储去掉重复的实体
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append((e1, relation, e2))
    # print("numer of unique_entities->", len(unique_entities))  # 不重复的实体有14951个
    return triples_data

# 加载训练数据
filename = "./data/FB15k-237/train.txt"
test_data = load_data(filename)  # 加载了训练的三元组
test_loader = DataLoader(dataset=test_data ,batch_size=128,shuffle=False,num_workers=0,drop_last=False)

filename1 = "./data/FB15k-237/valid.txt"
valid_data = load_data(filename1)  # 加载了训练的三元组
valid_loader = DataLoader(dataset=valid_data ,batch_size=128,shuffle=True,num_workers=0,drop_last=False)

filename2 = "./data/FB15k-237/test.txt"
test_data = load_data(filename2)  # 加载了训练的三元组
test_loader = DataLoader(dataset=test_data ,batch_size=128,shuffle=False,num_workers=0,drop_last=False)

# data是128个三元组，[0] 是头 [1是关系]
# for data in test_loader:
#     #print(data[0])
#     #print(len(data[1]))
#     pass


# 用来解析含有Trhead和Trtail的那个文件的，返回的第一个值是一个字典，第二个是一个列表，存储了所有的关系
def parse_type_text(type_filename):   # 取出418文件的类型集合和关系集合
    with open(type_filename) as f:
        type_dic = {}  # 最后的字典
        lines = f.read().split("----------------------")
        relation_list = []  # 字典的键
        typelist = []  # 字典的值
        relationlist = []
        for line in lines:
            typelist1 = []  # 包含头部类型集合 和尾部类型集合
            head_type_list = []  # 存储头部类型的集合
            tail_type_list = []  # 存储尾部类型的集合
            head_type_list1 = []  # 存储头部类型的集合
            tail_type_list1 = []  # 存储尾部类型的集合

            relation = line.split("head_type:")[0].strip("\n").strip("relation:").strip("\n  ")
            type_list = line.split("head_type:")[1]
            head_type = type_list.split("tail_type:")[0]  # 头部类型的字符串

            tail_type = type_list.split("tail_type:")[1]  # 尾部类型的字符串

            head_type_list = head_type.strip().strip("\t").split("\n")
            for m in head_type_list:
                head_type_list1.append(m.strip().strip("\n"))
            tail_type_list = tail_type.strip().strip("\t").split("\n")
            for n in tail_type_list:
                tail_type_list1.append(n.strip().strip("\n"))

            typelist1.append(head_type_list1)
            typelist1.append(tail_type_list1)
            typelist.append(typelist1)
            relationlist.append(relation)
            type_dic[relation] = typelist1

    return type_dic,relationlist

# 这个函数解析了实体和类型集对应的文件，返回的第一个值是一个字典，键是实体，值是实体的类型列表 ，返回的第二个值是实体的列表
def parse_entity2type(typeSrc):
    with open(typeSrc) as f:
        entity_list = []
        type_list = []
        entity_type_dic = {}  # 最后返回的字典
        text = f.readlines()
        for line in text:
            type = []
            entity = line.strip().split('\t')[0]
            type = line.strip().split('\t')[1:]
            entity_list.append(entity)
            type_list.append(type)
            entity_type_dic[entity] = type
    return entity_type_dic,entity_list


# 输入的是Trlist就是基于关系的类型的列表，batch_size是128，还有实体的类型
# 返回两个语义相似度，基于2017年的方法计算了头和关系  关系和尾巴 头和尾巴的语义相似度
'''
def calculate_S(TrList, batchSize, entityType):
    S = np.empty([3, 128], dtype=float)  # 计算Sheadr Stailr SheadTail 通过这个存储这三个 128*3
    prior = np.empty([128, 1], dtype=float)
    entity_heads = []
    entity_tails = []
    i = 0
    for i in range(0, batchSize):
        triple_data_batch = triple_data[i]
        entity_heads = triple_data_batch[0]
        entity_tails = triple_data_batch[2]
        relation = triple_data_batch[1]  # 取到batchSize里面所拥有的关系

        # 然后现在是要根据关系来计算Tr_head和Tr_tail
        # 键是关系，值是头部类型集合和尾部类型集合
        Trhead = TrList.get(relation)[0]
        Trtail = TrList.get(relation)[1]

        # 获取batch_size中头实体和 尾部实体对应的类型集和 就是Thead 和 Ttail
        if (entityType.get(entity_heads) != None) & (entityType.get(entity_tails) != None):
            Thead = entityType.get(entity_heads)
            Ttail = entityType.get(entity_tails)
            count0 = 0
            count1 = 0
            count2 = 0
            # print("这是第{}个实体{}".format(i,entity_heads))

            # Thead里面有Nonetype
            for t in Trhead:
                if (t in Thead):
                    count0 = count0 + 1
            for h in Ttail:
                if (h in Trtail):
                    count1 = count1 + 1
            for hx in Thead:
                if (hx in Ttail):
                    count2 = count2 + 1

            S[0][i] = count0 / len(Trhead)
            S[1][i] = count1 / len(Trtail)
            S[2][i] = count2 / len(Thead)
            prior[i] = S[0][i] * S[1][i]
        else:
            S[0][i] = 100
            S[1][i] = 100
            S[2][i] = 100
            prior[i] = 100
    return S, prior
'''

# 解析了两个文件
trSrc = "./data/FB15k-237/head_tail_type_train_new.txt"
typeSrc = "./data/FB15k-237/entity2type.txt"

type_list,relations_list = parse_type_text(trSrc)  # 获得了每个关系对应的Trhead 和 TrTail 还有关系列表
entity2type,entityList = parse_entity2type(typeSrc)   # 解析了每个实体对应的类型的列表 还有实体的列表
