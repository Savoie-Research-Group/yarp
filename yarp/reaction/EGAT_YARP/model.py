import torch.nn.functional as F
import torch
from torch.nn import Linear, Dropout
import torch.nn as nn
import dgl
import json
try:
    from egat import EGATConv
except ImportError:
    from yarp.reaction.EGAT_YARP.egat import EGATConv
try:
    from egat import EGATConv
except ImportError:
    from yarp.reaction.EGAT_YARP.egat import EGATConv

class EGAT_Rxn(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # parse input parameters
        num_node_feats, num_edge_feats, self.hidden_dim, self.num_heads = cfg.num_node_feats, cfg.num_edge_feats, cfg.hidden_dim, cfg.num_heads

        # BLOCK1: massage-passing blocks
        self.egat1 = EGATConv(in_node_feats=num_node_feats,in_edge_feats=num_edge_feats,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)
        self.egat2 = EGATConv(in_node_feats=self.hidden_dim*self.num_heads,in_edge_feats=self.hidden_dim*self.num_heads,out_node_feats=self.hidden_dim,out_edge_feats=self.hidden_dim,num_heads=self.num_heads)

        # BLOCK2: aggregate reactant and product nodes features
        self.agg_N_feats = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads, bias=True), nn.GELU())
        self.agg_E_feats = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads, self.hidden_dim*self.num_heads, bias=True), nn.GELU())

        # BLOCK3: final MLP layers
        self.mlp1 = nn.Sequential(nn.Linear(self.hidden_dim*self.num_heads*2, 256, bias=True),nn.GELU())
        self.mlp2 = nn.Sequential(nn.Linear(256, 128, bias=True),nn.GELU())
        self.mlp3 = nn.Linear(128, 1, bias=True)
        
        # print('Set: ',num_node_feats,num_edge_feats)

    def forward(self, graphR, graphP):

        ##################################### 
        ############# layer one ############# 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat1(graphR, graphR.ndata['x'], graphR.edata['x'])
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat1(graphP, graphP.ndata['x'], graphP.edata['x'])
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

        ##################################### 
        ############# layer two ############# 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)
        
        ##################################### 
        ############ layer three ############ 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

        ##################################### 
        ############ layer four ############# 
        ##################################### 
        Rnode_feats, Redge_feats = self.egat2(graphR, Rnode_feats, Redge_feats)
        Rnode_feats = Rnode_feats.view(graphR.number_of_nodes(),self.hidden_dim * self.num_heads)
        Redge_feats = Redge_feats.view(graphR.number_of_edges(),self.hidden_dim * self.num_heads)

        Pnode_feats, Pedge_feats = self.egat2(graphP, Pnode_feats, Pedge_feats)
        Pnode_feats = Pnode_feats.view(graphP.number_of_nodes(),self.hidden_dim * self.num_heads)
        Pedge_feats = Pedge_feats.view(graphP.number_of_edges(),self.hidden_dim * self.num_heads)

        # merge R and P features
        Rxn_node_feature = self.agg_N_feats(Pnode_feats - Rnode_feats)
        Rxn_edge_feature = self.agg_E_feats(Pedge_feats - Redge_feats)

        # obtain global feature for larer 3
        graphR.ndata['x'] = Rxn_node_feature
        graphR.edata['x'] = Rxn_edge_feature 
        individual_graphs = dgl.unbatch(graphR)

        # Initialize a list to store the global features for each graph
        G_node_feats,G_edge_feats = [],[]

        # Iterate through the individual graphs
        for graph in individual_graphs:
            # Calculate the sum of the node features (assuming node features are stored in 'h')
            global_node_feature = graph.ndata['x'].sum(dim=0)
            global_edge_feature = graph.edata['x'].sum(dim=0)
            G_node_feats.append(global_node_feature)
            G_edge_feats.append(global_edge_feature)

        G_node_feats = torch.stack(G_node_feats)
        G_edge_feats = torch.stack(G_edge_feats)

        #################################### 
        ########## merge features ##########
        #################################### 
        G_features = torch.cat((G_node_feats,G_edge_feats), axis=1)

        # MLP
        x   = self.mlp1(G_features)
        x   = self.mlp2(x)
        x   = self.mlp3(x)

        return x
