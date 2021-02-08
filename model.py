import torch
import torch.nn as nn
from copy import deepcopy
    
    
class GRUPredictor(nn.Module):
    
    def __init__(self, num_timeslots, time_embedding_dim, num_clusters, cluster_embedding_dim, hidden_dim, latent_dim=256, n_layers=1):
        
        super(GRUPredictor, self).__init__()
        
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_embedding_dim)
        self.time_embedding = nn.Embedding(num_timeslots, time_embedding_dim)
        
        self.gru = nn.GRU(cluster_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, num_clusters)
        )
        
        self.criteria = nn.CrossEntropyLoss()
        
    def get_hidden_state(self, xc, xt):
        merged = torch.cat([self.cluster_embedding(xc), self.time_embedding(xt)], dim=2)
        output, _ = self.gru(merged, None)
        return output
    
    def forward(self, xc, xt, hidden=None):
        if hidden is None:
            hidden = self.get_hidden_state(xc, xt)
        output = self.out(hidden[:, -1])
        return output
    
    
class GRUPredictorCondition(nn.Module):
    
    def __init__(self, num_timeslots, time_embedding_dim, num_clusters, cluster_embedding_dim, hidden_dim, num_weekday=7, latent_dim=256, n_layers=1):
        
        super(GRUPredictorCondition, self).__init__()
        
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_embedding_dim)
        self.time_embedding = nn.Embedding(num_timeslots, time_embedding_dim)
        self.weekday_embedding = nn.Embedding(num_weekday, time_embedding_dim)
        
        self.gru = nn.GRU(cluster_embedding_dim + 2 * time_embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(latent_dim, num_clusters)
        )
        
        self.criteria = nn.CrossEntropyLoss()
        
    def get_hidden_state(self, xc, xt, xd):
        merged = torch.cat([self.cluster_embedding(xc), self.time_embedding(xt), self.weekday_embedding(xd)], dim=2)
        output, _ = self.gru(merged, None)
        return output
    
    def forward(self, xc, xt, xd, hidden=None):
        if hidden is None:
            hidden = self.get_hidden_state(xc, xt, xd)
        output = self.out(hidden[:, -1])
        return output
    
    
class CrowdPredictor(nn.Module):
    
    def __init__(self, num_timeslots, time_embedding_dim, num_clusters, cluster_embedding_dim, hidden_dim, context_dim, latent_dim=256, n_layers=1, pooling_func='mean'):
        
        super(CrowdPredictor, self).__init__()
        
        self.context_dim = context_dim 
        
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_embedding_dim)
        self.time_embedding = nn.Embedding(num_timeslots, time_embedding_dim)
        
        self.gru = nn.GRU(cluster_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers)
        
        self.context_net = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.Tanh(),
        )
        
        self.pooling_func = pooling_func
        
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, num_clusters)
        )
        
    def get_context(self, xc, xt):
        merged = torch.cat([self.cluster_embedding(xc), self.time_embedding(xt)], dim=-1)
        output = self.gru(merged, None)[0][:, -1]
        if self.pooling_func == 'mean':
            return self.context_net(output).mean(dim=0, keepdim=True)
        else:
            return self.context_net(output).max(dim=0, keepdim=True)
        
    def forward(self, xc, xt, precompute_context=None):
        merged = torch.cat([self.cluster_embedding(xc), self.time_embedding(xt)], dim=-1)
        output = self.gru(merged, None)[0][:, -1]
        if precompute_context is None:
            if self.pooling_func == 'mean':
                context = self.context_net(output).mean(dim=0, keepdim=True)
            else:
                context = self.context_net(output).max(dim=0, keepdim=True).values
        else:
            context = precompute_context
        return self.out_net(torch.cat([output, context.repeat(xc.shape[0], 1)], dim=-1))


class CrowdPredictorGRU(nn.Module):
    
    def __init__(self, num_timeslots, time_embedding_dim, num_clusters, cluster_embedding_dim, hidden_dim, context_dim, latent_dim=256, n_layers=1):
        
        super(CrowdPredictorGRU, self).__init__()
        
        self.context_dim = context_dim 
        
        self.cluster_embedding = nn.Embedding(num_clusters, cluster_embedding_dim)
        self.time_embedding = nn.Embedding(num_timeslots, time_embedding_dim)
        
        self.gru = nn.GRU(cluster_embedding_dim + time_embedding_dim, hidden_dim, batch_first=True, num_layers=n_layers)
        
        self.context_net_avg = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.Tanh(),
        )
        self.context_gru = nn.GRU(context_dim, context_dim, batch_first=True, num_layers=n_layers)
        
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, num_clusters)
        )
        
    def get_context(self, xc, xt):
        merged = torch.cat([self.cluster_embedding(xc), self.time_embedding(xt)], dim=-1)
        output = self.gru(merged, None)[0]
        return self.context_net_avg(output).mean(dim=0, keepdim=True)
        
    def forward(self, xc, xt, precompute_context=None):
        merged = torch.cat([self.cluster_embedding(xc), self.time_embedding(xt)], dim=-1)
        output = self.gru(merged, None)[0]
        if precompute_context is None:
            context = self.context_net_avg(output).mean(dim=0, keepdim=True)
        else:
            context = precompute_context
        return self.out_net(torch.cat([output[:, -1], self.context_gru(context, None)[0][:, -1].repeat(xc.shape[0], 1)], dim=-1))
    
    
class EnsemblePredictor(nn.Module):
    
    def __init__(self, num_clusters, cluster_embedding_dim, n_components):
        
        super(EnsemblePredictor, self).__init__()
        
        self.multiplex = nn.Sequential(
            nn.Embedding(num_clusters, cluster_embedding_dim),
            nn.Linear(cluster_embedding_dim, n_components),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, xc, preds):
        weights = self.multiplex(xc)
        return torch.einsum('bk,kbi->bi', weights, preds)