# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu
import torch.nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# num_node_features = 1
num_node_features = 10*6
hidden = 64

class GCN_test(torch.nn.Module):
    def __init__(self):
        super(GCN_test, self).__init__()
        # self.conv1 = GCNConv(num_node_features, 1)
        # self.conv2 = GCNConv(hidden, hidden)
        # self.conv3 = GCNConv(hidden, hidden)
        # self.lin = Linear(hidden, 1)
        # self.lin = Linear(hidden, 5)
        #
        # self.lin1 = Linear(5+2, 5)
        self.lin2 = Linear(4,3)

    # def forward(self,x, edge_index, batch, dist, sw):
    #     x= self.conv1(x, edge_index)
    #     x= x.relu()
    #     x = self.conv2(x, edge_index)
    #     x = x.relu()
    #     x = self.conv3(x, edge_index)
    #     x = global_mean_pool(x, batch)
    #     x = F.dropout(x, p=0.5,training=self.training)
    #     x = self.lin(x)
    #     # print(x.shape)
    #     # print(sw.shape)
    #     x = self.lin1(torch.cat((x,dist,sw), dim=1))
    #     x = x.relu()
    #     x = self.lin2(x)
    #     return x

    def forward(self, x):
        # x = self.conv1(x, edge_index)

        x = torch.flatten(x)


        x = self.lin2( x)



        # x = torch.unsqueeze(x,0)
        return x

















