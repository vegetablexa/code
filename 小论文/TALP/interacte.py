from helper import *
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from data_loader import *
from model import *
import LVJIANNAN_________prior as P
import LVJIANNAN___model as M


class Main(object):

	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model
		
		Returns
		-------
		Creates computational graph and optimizer
		
		"""
		self.p = params
		self.logger = get_logger(self.p.name, self.p.log_dir, self.p.config_dir)

		self.logger.info(vars(self.p))
		pprint(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		else:
			self.device = torch.device('cuda:1')

		self.load_data()

		self.model        = self.add_model()
		self.optimizer    = self.add_optimizer(self.model.parameters())

	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format. 

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset  (FB15k-237, WN18RR, YAGO3-10)
		
		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits
		self.chequer_perm:      Stores the Chequer reshaping arrangement

		"""

		ent_set, rel_set = OrderedSet(), OrderedSet()
		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				ent_set.add(sub)
				rel_set.add(rel)
				ent_set.add(obj)

		self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
		self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}

		self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

		# print(self.rel2id)
		self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
		self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
		self.p.num_ent		= len(self.ent2id)
		self.p.num_rel		= len(self.rel2id) // 2
		#self.p.num_rel = len(self.rel2id)
		self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim


		self.data	= ddict(list)
		sr2o		= ddict(set)

		for split in ['train', 'test', 'valid']:
			for line in open('./data/{}/{}.txt'.format(self.p.dataset, split)):
				sub, rel, obj = map(str.lower, line.strip().split('\t'))
				sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
				self.data[split].append((sub, rel, obj))

				if split == 'train': 
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)
		self.data = dict(self.data)

		self.sr2o = {k: list(v) for k, v in sr2o.items()}
		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				sr2o[(sub, rel)].add(obj)
				sr2o[(obj, rel+self.p.num_rel)].add(sub)

		self.sr2o_all = {k: list(v) for k, v in sr2o.items()}

		self.triples = ddict(list)

		if self.p.train_strategy == 'one_to_n':
			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
		else:
			for sub, rel, obj in self.data['train']:
				rel_inv		= rel + self.p.num_rel
				sub_samp	= len(self.sr2o[(sub, rel)]) + len(self.sr2o[(obj, rel_inv)])
				sub_samp	= np.sqrt(1/sub_samp)

				self.triples['train'].append({'triple':(sub, rel, obj),     'label': self.sr2o[(sub, rel)],     'sub_samp': sub_samp})
				self.triples['train'].append({'triple':(obj, rel_inv, sub), 'label': self.sr2o[(obj, rel_inv)], 'sub_samp': sub_samp})

		for split in ['test', 'valid']:
			for sub, rel, obj in self.data[split]:
				rel_inv = rel + self.p.num_rel
				self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
				self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

		self.triples = dict(self.triples)

		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)

		self.data_iter = {
			'train'		:   get_data_loader(TrainDataset, 'train', 	self.p.batch_size),
			'valid_head'	:   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail'	:   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head'	:   get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail'	:   get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
			# 'test_head': get_data_loader(TestDataset, 'test_head', 32),
			# 'test_tail': get_data_loader(TestDataset, 'test_tail', 32),
		}

		self.chequer_perm	= self.get_chequer_perm()

	def get_chequer_perm(self):
		"""
		Function to generate the chequer permutation required for InteractE model

		Parameters
		----------
		
		Returns
		-------
		
		"""
		ent_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])
		rel_perm  = np.int32([np.random.permutation(self.p.embed_dim) for _ in range(self.p.perm)])

		comb_idx = []
		for k in range(self.p.perm):
			temp = []
			ent_idx, rel_idx = 0, 0

			for i in range(self.p.k_h):
				for j in range(self.p.k_w):
					if k % 2 == 0:
						if i % 2 == 0:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
						else:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
					else:
						if i % 2 == 0:
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
						else:
							temp.append(ent_perm[k, ent_idx]); ent_idx += 1;
							temp.append(rel_perm[k, rel_idx]+self.p.embed_dim); rel_idx += 1;

			comb_idx.append(temp)

		chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
		return chequer_perm


	def add_model(self):
		"""
		Creates the computational graph

		Parameters
		----------
		
		Returns
		-------
		Creates the computational graph for model and initializes it
		
		"""
		model = InteractE(self.p, self.chequer_perm)
		model.to(self.device)
		return model

	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model
		
		Returns
		-------
		Returns an optimizer for learning the parameters of the model
		
		"""
		if self.p.opt == 'adam': return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)
		else:			 return torch.optim.SGD(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split

		
		Returns
		-------
		triples:	The triples used for this split
		labels:		The label for each triple
		"""
		if split == 'train':
			if self.p.train_strategy == 'one_to_x':
				triple, label, neg_ent, sub_samp = [ _.to(self.device) for _ in batch]
				return triple[:, 0], triple[:, 1], triple[:, 2], label, neg_ent, sub_samp
			else:
				triple, label = [ _.to(self.device) for _ in batch]
				return triple[:, 0], triple[:, 1], triple[:, 2], label, None, None
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
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
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p)
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model
		
		Returns
		-------
		"""
		state				= torch.load(load_path)
		state_dict			= state['state_dict']
		self.best_val_mrr 		= state['best_val']['mrr']
		self.best_val 			= state['best_val']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])

	def evaluate(self, split, epoch=0):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""		
		left_results  = self.predict(split=split, mode='tail_batch')
		right_results = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)
		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
		return results



	# 先验模型的部分 存储main函数的部分
	def prior_calcute_tail(self, sub, rel, tail):
		device = torch.device('cuda' if torch.cuda.is_available() else "cuda:1")

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
		label = []
		test_matrix = []
		for i in range(0, len(rel)):
			# 传进来的关系是按照interacte中的关系来定义的，
			# if tail[i] not in P.entityList:
			# 	label.append(0)
			# else:
			# 	label.append(P.entityList.index(tail[i]))
			relindex.append(rel[i].item())
			#label.append(P.entityList.index(tail[i]))
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
		# predictlabel = torch.argmax(pred, 1)
		# preds = model.deal(pred, label, predictlabel)

		return pred

	# 根据尾部实体来预测头
	def prior_calcute_head(self, sub, rel, tail):
		device = torch.device('cuda' if torch.cuda.is_available() else "cuda:1")

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
		label = []
		for i in range(0, len(rel)):
			# 传进来的关系是按照interacte中的关系来定义的，
			# if head[i] not in P.entityList:
			# 	label.append(0)
			# else:
			# 	label.append(P.entityList.index(head[i]))
			relll = rel[i] - 237

			relindex.append(relll.item())

			index = relll.item()
			# print("先验概率算的关系是那个？")
			# print(P.relationList[index])
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
		# predictlabel = torch.argmax(pred, 1)
		# preds = model.deal(pred, label, predictlabel)

		return pred

	# 构建了训练用的矩阵
	def prior_train_matrix(self):
		self.matrix_train = np.empty([len(self.rel2id), len(P.type_List)])
		# print(len(P.type_List))
		# print(len(P.typeList))
		# print(self.matrix_train.shape)
		# 474 * 1590
		# print(matrix.shape)
		# 每一行

		for i in range(0, len(P.relationList)):
			# 每一列
			Trhead = P.deal_relationlist.get(P.relationList[i])[0]
			# print(Trhead)
			Trtail = P.deal_relationlist.get(P.relationList[i])[1]
			for type in Trhead.keys():
				index = P.type_List.index(type)
				# print(Trhead.get(type))
				self.matrix_train[i+237][index] = Trhead.get(type)
			for type in Trtail.keys():
				index = P.type_List.index(type)
				# print(Trhead.get(type))
				self.matrix_train[i][index] = Trtail.get(type)
			# matrix[rel][type] = round(Trtail.get(type), 2)
		# print(matrix)
		# print(matrix1)
		return self.matrix_train

	# 训练用的函数
	def prior_calcute_train(self,sub, rel, tail):
		device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		matrix1 = self.prior_train_matrix()
		# ##？？？？ 按照新的方法先对实体和关系编号，然后构建新的实体01矩阵和关键矩阵
		matrix = torch.Tensor(matrix1).to(device)

		# 传入的是头的权重矩阵和 尾的权重矩阵
		# 实体类型的01矩阵 14951*1590
		matrix_entity_01 = torch.Tensor(P.matrix_entity).to(device)
		model = M.model(matrix, matrix_entity_01).to(device)

		head = sub
		rel = rel
		tail = tail
		# 每一个batch的标签  也就是真实值
		relindex = []

		test_matrix = []
		for i in range(0, len(rel)):
			# 传进来的关系是按照interacte中的关系来定义的，
			relindex.append(rel[i].item())
			index = rel[i]
			test_matrix.append(matrix1[index])

		# numpy强制类型转换
		test_matrix = np.matrix(test_matrix)
		# print("每一批的训练数据")
		# print(test_matrix)
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


	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
		mode: (string):		Can be 'head_batch' or 'tail_batch'
		
		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)

				#
				# if mode == 'tail_batch':
				#
				# 	prior = self.prior_calcute_tail(sub,rel,obj)
				#
				# else:
				# 	# 把每个反关系的index调整后得到了正关系 然后根据关系预测头部实体的概率
				#
				# 	prior = self.prior_calcute_head(sub,rel,obj)

				# 给定尾巴和关系来预测头
				pred			= self.model.forward(sub, rel, None, 'one_to_n')
				#
				# pred = (prior * pred).to(self.device)

				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				pred 			= torch.where(label.byte(), torch.zeros_like(pred), pred)
				pred[b_range, obj] 	= target_pred
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]

				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0)
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0)
				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))

		return results


	def run_epoch(self, epoch):
		"""
		Function to run one epoch of training

		Parameters
		----------
		epoch: current epoch count
		
		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, batch in enumerate(train_iter):
			self.optimizer.zero_grad()

			sub, rel, obj, label, neg_ent, sub_samp = self.read_batch(batch, 'train')


			# 在此处加入先验概率 在进行反向传播

			pred	= self.model.forward(sub, rel, neg_ent ,self.p.train_strategy)
			# print(pred)

			prior = self.prior_calcute_train(sub, rel, obj)
			# print(pred)
			# pred.detach().cpu().numpy()[np.isnan(pred.detach().cpu().numpy())] = 0
			prior = prior.detach().cpu().numpy()

			prior[np.isnan(prior)] = 0.0000000000000000001
			prior = torch.tensor(prior).to(self.device)
			prior = torch.sigmoid(prior)
			pred = (pred * prior).to(self.device)
			# pred = torch.tensor(pred).to(self.device)
			# print(np.where(pred.detach().cpu().numpy()<0))
			# print(np.where(pred.detach().cpu().numpy() > 1))
			# print(np.where(np.isnan(pred.detach().cpu().numpy()) == True))
			# print(pred)

			loss	= self.model.loss(pred, label, sub_samp)


			# 训练相乘后 loss 不好用了  sigmodi
			# print(loss)
			# print("打印损失值")
			# print(loss)
			loss.backward()
			self.optimizer.step()
			losses.append(loss.item())

			if step % 100 == 0:
				self.logger.info('[E:{}| {}]: Train Loss:{:.5},  Val MRR:{:.5}, \t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))

		loss = np.mean(losses)
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss

	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		----------
		
		Returns
		-------
		# """
		# self.best_val_mrr, self.best_val, self.best_epoch = 0., {}, 0.
		# val_mrr = 0
		#save_path = os.path.join('./torch_saved', self.p.name)
		# # testrun_7cbab46f   testrun_62386ae7
		# testrun_3daded9e

		#
		# save_path = os.path.join('./torch_saved', "arctan+weichuli.crdownload")
		save_path = "D:\\luoenze\\arctan+weichuli.crdownload"
		# self.load_model(save_path)
		# if self.p.restore:
		# 	self.load_model(save_path)
		# 	self.logger.info('Successfully Loaded previous model')
		#
		# for epoch in range(self.p.max_epochs):
		# 	train_loss	= self.run_epoch(epoch)
		# 	# val_results	= self.evaluate('valid', epoch)
		# 	#
		# 	# if val_results['mrr'] > self.best_val_mrr:
		# 	# 	self.best_val		= val_results
		# 	# 	self.best_val_mrr	= val_results['mrr']
		# 	# 	self.best_epoch		= epoch
		# 	# 	self.save_model(save_path)
		# 	# self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Valid MRR: {:.5}, \n\n\n'.format(epoch, train_loss, self.best_val_mrr))
		#
		# 	val_results = self.evaluate('test', epoch)
		#
		# 	if val_results['mrr'] > self.best_val_mrr:
		# 		self.best_val = val_results
		# 		self.best_val_mrr = val_results['mrr']
		# 		self.best_epoch = epoch
		# 		self.save_model(save_path)
		# 	self.logger.info('[Epoch {}]:  Training Loss: {:.5},  Test MRR: {:.5}, \n\n\n'.format(epoch, train_loss,
		# 																						   self.best_val_mrr))

		# Restoring model corresponding to the best validation performance and evaluation on test data
		self.logger.info('Loading best model, evaluating on test data')
		self.load_model(save_path)
		self.evaluate('test')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	# Dataset and Experiment name
	parser.add_argument('--data',           dest="dataset",         default='FB15k-237',            		help='Dataset to use for the experiment')
	parser.add_argument("--name",            			default='testrun_'+str(uuid.uuid4())[:8],	help='Name of the experiment')

	# Training parameters
	parser.add_argument("--gpu",		type=str,               default='0',					help='GPU to use, set -1 for CPU')
	parser.add_argument("--train_strategy", type=str,               default='one_to_n',				help='Training strategy to use')
	parser.add_argument("--opt", 		type=str,               default='adam',					help='Optimizer to use for training')
	parser.add_argument('--neg_num',        dest="neg_num",         default=1000,    	type=int,       	help='Number of negative samples to use for loss calculation')
	parser.add_argument('--batch',          dest="batch_size",      default=128,    	type=int,       	help='Batch size')
	parser.add_argument("--l2",		type=float,             default=0.0,					help='L2 regularization')
	parser.add_argument("--lr",		type=float,             default=0.00001,					help='Learning Rate')
	parser.add_argument("--epoch",		dest='max_epochs', 	default=3000,		type=int,  		help='Maximum number of epochs')
	parser.add_argument("--num_workers",	type=int,               default=0,                      		help='Maximum number of workers used in DataLoader')
	parser.add_argument('--seed',           dest="seed",            default=42,   		type=int,       	help='Seed to reproduce results')
	parser.add_argument('--restore',   	dest="restore",       	action='store_true',            		help='Restore from the previously saved model')

	# Model parameters
	parser.add_argument("--lbl_smooth",     dest='lbl_smooth',	default=0.1,		type=float,		help='Label smoothing for true labels')
	parser.add_argument("--embed_dim",	type=int,              	default=None,                   		help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
	parser.add_argument('--bias',      	dest="bias",          	action='store_true',            		help='Whether to use bias in the model')
	parser.add_argument('--form',		type=str,               default='plain',            			help='The reshaping form to use')
	parser.add_argument('--k_w',	  	dest="k_w", 		default=10,   		type=int, 		help='Width of the reshaped matrix')
	parser.add_argument('--k_h',	  	dest="k_h", 		default=20,   		type=int, 		help='Height of the reshaped matrix')
	parser.add_argument('--num_filt',  	dest="num_filt",      	default=96,     	type=int,       	help='Number of filters in convolution')
	parser.add_argument('--ker_sz',    	dest="ker_sz",        	default=9,     		type=int,       	help='Kernel size to use')
	parser.add_argument('--perm',      	dest="perm",          	default=1,      	type=int,       	help='Number of Feature rearrangement to use')
	parser.add_argument('--hid_drop',  	dest="hid_drop",      	default=0.5,    	type=float,     	help='Dropout for Hidden layer')
	parser.add_argument('--feat_drop', 	dest="feat_drop",     	default=0.5,    	type=float,     	help='Dropout for Feature')
	parser.add_argument('--inp_drop',  	dest="inp_drop",      	default=0.2,    	type=float,     	help='Dropout for Input layer')

	# Logging parameters
	parser.add_argument('--logdir',    	dest="log_dir",       	default='./log/',               		help='Log directory')
	parser.add_argument('--config',    	dest="config_dir",    	default='./config/',            		help='Config directory')
	

	args = parser.parse_args()

	torch.set_num_threads(4)

	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Main(args)
	model.fit()
