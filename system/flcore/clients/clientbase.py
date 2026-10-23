"""Base client (trimmed from PFLlib's flcore/clients/clientbase.py; only what FedLoRD reaches)."""
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.local_iters = getattr(args, 'local_iters', -1)   # K updates/round (>0) or legacy epochs (-1)
        self.few_shot = args.few_shot

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        # plain SGD (no momentum) with weight decay on the shared part, as in the paper protocol
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         momentum=0.0,
                                         weight_decay=getattr(args, 'weight_decay', 1e-3))
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        loader = DataLoader(train_data, batch_size, drop_last=True, shuffle=True)
        if len(loader) == 0:
            # client holds fewer than batch_size samples: keep the partial batch,
            # otherwise the loader is empty and a K-step loop could never terminate.
            loader = DataLoader(train_data, batch_size, drop_last=False, shuffle=True)
        return loader

    def iter_batches(self, loader, K):
        """Yield exactly K mini-batches, cycling (re-shuffling) the loader as often as needed."""
        n = 0
        while n < K:
            for batch in loader:
                yield batch
                n += 1
                if n >= K:
                    return

    def local_steps(self):
        """K for this round under the equal-update protocol (-li>0); None under legacy epoch mode."""
        if self.local_iters is None or self.local_iters <= 0:
            return None
        K = self.local_iters
        if self.train_slow:
            K = int(np.random.randint(1, max(2, K // 2)))
        return K

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
