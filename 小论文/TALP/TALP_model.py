
import torch.nn as nn
import torch.nn.functional as F
import torch


device = torch.device('cuda' if torch.cuda.is_available() else "cpu") #******#
class model(nn.Module):
    # 传入的是实体的尾部权重矩阵 和 实体的类型01矩阵
    def __init__(self,matrix_Trtail,matrix_entity):
        super(model, self).__init__()
        #self.bceloss = F.binary_cross_entropy()
        self.matrix_Trtail = matrix_Trtail.to(device)
        self.W2 = nn.Parameter((matrix_Trtail).to(device))

        # 传入实体的01类型 14941*1590
        self.matrix_ent = matrix_entity.to(device)

    def forward(self,fenmu,rel):
        test_index = torch.tensor(rel).to(device)

        # 求一下分子
        #test_matrix = torch.index_select(self.W2,dim=0,index = test_index).to(device)

        test_matrix = self.W2[rel]
#        test_matrix = torch.Tensor(test_matrix)
#        test_matrix = test_matrix.to(torch.float32)
#        test_matrix.to(device)
#         print("取出指定行的参数标进行反向")
#         print(test_matrix)
        x = torch.mm(test_matrix.to(device), self.matrix_ent.T.to(device))
        x.to(device)
        fenmu = fenmu.to(device)
        # print(x.shape)
        # print(fenmu.shape)
        #print(self.W2.sum(1))
        #print("在计算先验概率呢")
        return x/fenmu

    def loss(self,pred,true_label,label,predictlabel):
        count = 0
        for i in range(0, pred.shape[0]):
            # 没有命中 算loss
            # if(true_label[i] != pred[i]):
            zhenshigailv = pred[i][label[i]]
            yucegailv = pred[i][predictlabel[i]]

            if (zhenshigailv == yucegailv):
                count = count+1
                # if(preds[i][true_label]==preds[i][pred[i]]):

                # count = count + 1
        mingzhong = ((count) / pred.shape[0]) * 100
        loss = F.binary_cross_entropy(pred,true_label)
        return loss,mingzhong

    def deal(self,pred,label,predictlabel):
        b = torch.zeros([pred.shape[1]]).to(device)
        for i in range(0, pred.shape[0]):
            zhenshigailv = pred[i][label[i]]
            yucegailv = pred[i][predictlabel[i]]
            if (zhenshigailv == yucegailv):
                pass
            else:
                c = pred[i]
                pred[i] = torch.where(c>zhenshigailv,b,c)
        return pred



