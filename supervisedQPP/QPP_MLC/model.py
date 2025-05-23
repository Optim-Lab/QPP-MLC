#%%
import os
import glob
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AlbertModel, RobertaModel, ElectraModel, DebertaV2Model, T5EncoderModel

#%%
class QPP_MLC(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.min_k_m = min(self.args.top_m, self.args.top_k)

        # BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.d_model = self.bert.config.hidden_size
    
        # Transformer
        if self.args.transformer:
            self.transformer = TransformerForQPP(
                d_model=self.d_model,
                nhead=self.args.trans_nhead,
                num_layers=self.args.trans_num_layers
                )

        # FFNN (also called MLP)
        self.mlp = MLP(input_dim=self.d_model, output_dim=1, mlp_bias=True)

        # loss
        self.loss_MSE = torch.nn.MSELoss(reduction='mean')
        self.loss_BCE = torch.nn.BCELoss(reduction='none')


    def forward(self, data, debug=False, target_metric='mrr@10'):
        # assert data["input_ids"].size()[1] == self.args.top_k
        assert data["input_ids"].size()[2] == 256
        batch_size = data["input_ids"].size()[0]
        qid_list = data['qid']
        actual_relevance = data['rg']
        score = data['score']
        rr_true = data['rr'].float()
        idcg_true = data['idcg'].float()
        dcg_true = data['dcg'].float()
        ndcg_true = data['ndcg'].float()
        ap = data['ap']
        
        # bert = bert.cuda()
        # transformer = transformer.cuda()
        # mlp = mlp.cuda()
        # generate q and d embedding by BERT
        input_ids =           data["input_ids"][:,:self.args.top_k,].contiguous().view(batch_size * self.args.top_k, 256)
        attention_mask = data["attention_mask"][:,:self.args.top_k,].contiguous().view(batch_size * self.args.top_k, 256)
        token_type_ids = data["token_type_ids"][:,:self.args.top_k,].contiguous().view(batch_size * self.args.top_k, 256)

        embed = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            output_hidden_states=False,
            return_dict=True).last_hidden_state[:, 0]

        # change embeddings shape [batch_size, top_k, embedding_dim]
        # embed = torch.rand((16, 10, 768), device='cuda') # example tensor
        embed = embed.view(batch_size, self.args.top_k, self.d_model)
        
        # transformer output [batch, top_k, 768]
        if self.args.transformer:
            embed = self.transformer(embed)
            
        # predicted relevance using mlp [batch, top_k]
        # predicted_relevance = torch.rand((43,10), device='cuda') # example tensor
        predicted_relevance = self.mlp(embed).squeeze(2)

        if not self.args.err:
            predicted_relevance = self.threshold(predicted_relevance)

        # predicted dcg
        dcg_pred = self.compute_batch_dcg(predicted_relevance)
        
        # predicted idcg
        idcg_pred = self.compute_batch_idcg(predicted_relevance)
        
        # Compute NDCG
        ndcg_pred = dcg_pred / idcg_pred
        ndcg_pred[idcg_pred == 0] = 0  # Handle division by zero if IDCG is 0
        
        # Compute rr
        if self.args.err:
            rr_pred = self.compute_batch_err(predicted_relevance) # [batch]
        else:
            rr_pred = self.compute_batch_rr(predicted_relevance) # [batch]
        
        # predicted performance
        if target_metric == 'mrr@10':
            pp = rr_pred # [batch]

        elif target_metric == 'ndcg@10':
            pp = dcg_pred
        
        else:
            print('target metric should be mrr@10 or ndcg@10')
        
        # ndcg, when given idcg
        ndcg_tilde = dcg_pred / idcg_true
        ndcg_tilde[idcg_true == 0] = 0  # Handle division by zero if IDCG is 0

        if target_metric == 'mrr@10':
            loss = self.loss(
                predicted_relevance=predicted_relevance,
                actual_relevance=actual_relevance,
                posi_weight=self.args.posi_weight
                )
        else:
            loss = torch.tensor(0.0)

        if debug:
            dict_embed = {
                'embed' : embed,
                'qid_list' : torch.tensor([int(qid) for qid in qid_list]),
                'predicted_relevance' : predicted_relevance,
                'actual_relevance' : actual_relevance,
                'score' : score,
                'rr_pred' : rr_pred,
                'dcg_pred' : dcg_pred,
                'idcg_pred' : idcg_pred,
                'ndcg_pred' : ndcg_pred,
                'rr_true' : rr_true,
                'dcg_true' : dcg_true,
                'idcg_true' : idcg_true,
                'ndcg_true' : ndcg_true,
                'ndcg_tilde' : ndcg_tilde,
            }
            return loss, pp, dict_embed
        else:
            return loss, pp, ndcg_tilde


    def loss(
            self,
            predicted_relevance,
            actual_relevance,
            posi_weight,
            ):
        """
        Compute loss based on ERR-like weights for a batch of relevance scores.
        
        Args:
            relevance (torch.Tensor): Predicted relevance scores of shape (batch_size, k).
            relevance_true (torch.Tensor): True relevance scores of shape (batch_size, k).
        
        Returns:
            torch.Tensor: The total loss for the batch.
        """
        batch_size, top_k = predicted_relevance.shape
        actual_relevance_partial = actual_relevance[:, :top_k]
        
        # Binary Cross Entropy loss (per element)
        loss_element = self.loss_BCE(predicted_relevance.float(), actual_relevance_partial.float())
        
        # Compute cumulative products for (1 - relevance) for ERR weight calculation
        one_minus_relevance = 1 - predicted_relevance
        prod_cum = torch.cumprod(one_minus_relevance, dim=1)
        
        # Adjust cumulative products so that prod_cum[:, r] represents \prod_{j=1}^{r-1} (1 - relevance[:, j])
        prod_cum = torch.cat([torch.ones(batch_size, 1, device=predicted_relevance.device), prod_cum[:, :-1]], dim=1)
        
        # ERR weights: w_r = (1 / (r + 1)) * prod_cum[:, r]
        weights = prod_cum / torch.arange(1, top_k + 1, device=predicted_relevance.device, dtype=torch.float32)
        weights_normalized = weights / weights.sum(dim=1).unsqueeze(1)
        # weights_normalized.sum(dim=1) : 1
        
        # posi weight apply
        loss_element[actual_relevance_partial == 1] = posi_weight * loss_element[actual_relevance_partial == 1]
        
        # nomalize for posi_weight
        loss_element = loss_element * (1 + posi_weight) / (2 * posi_weight)
        
        # Weighted loss
        weights_one = torch.ones(prod_cum.shape, device=predicted_relevance.device) / top_k
        weighted_loss = weights_one * loss_element

        # calculate loss
        loss = weighted_loss.sum() / batch_size  # Normalize by batch size

        return loss


    def compute_batch_err(self, predicted_relevance):
        """
        Compute ERR@k for a batch of relevance scores without using loops.
        
        Args:
            relevance (torch.Tensor): Tensor of shape (batch_size, k), where each row
                                    contains relevance scores r_1, ..., r_k for a query.
        
        Returns:
            torch.Tensor: ERR@k for each query in the batch, shape (batch_size,).
        """
        predicted_relevance_partial = predicted_relevance[:,:self.min_k_m]
        batch_size, top_m = predicted_relevance_partial.shape
        
        # Compute cumulative product of (1 - r_j) for ERR weight calculation
        one_minus_relevance = 1 - predicted_relevance_partial
        prod_cum = torch.cumprod(one_minus_relevance, dim=1)
        
        # Adjust cumulative product so that prod_cum[:, i] represents \prod_{j=1}^{i-1} (1 - r_j)
        prod_cum = torch.cat([torch.ones(batch_size, 1, device=predicted_relevance.device), prod_cum[:, :-1]], dim=1)
        
        # Compute ERR: Sum over (1 / i) * r_i * prod_cum[:, i]
        weights = 1 / torch.arange(1, top_m + 1, device=predicted_relevance.device, dtype=torch.float32)
        err_pred = torch.sum(weights * predicted_relevance_partial * prod_cum, dim=1)
        
        return err_pred


    def compute_batch_rr(self, predicted_relevance):
        predicted_relevance_partial = predicted_relevance[:,:self.min_k_m]

        # Find the first occurrence of 1 in each row (along the top_m dimension)
        positions = torch.arange(1, self.min_k_m + 1, device=predicted_relevance.device).float()

        # Compute reciprocal rank for the first 1 in each row
        rr_pred = (predicted_relevance_partial / positions).max(dim=1).values
        
        return rr_pred
    

    def compute_batch_dcg(self, predicted_relevance):
        # Compute DCG
        discounts = torch.log2(torch.arange(2, self.args.top_k + 2).float().to(predicted_relevance.device))

        # gains = predicted_relevance[:, :10]
        # dcg_pred = (gains / discounts[:10]).sum(dim=-1)
        gains = predicted_relevance[:, :self.min_k_m]
        dcg_pred = (gains / discounts[:self.min_k_m]).sum(dim=-1)

        return dcg_pred
    

    def compute_batch_idcg(self, predicted_relevance):
        # Compute DCG
        discounts = torch.log2(torch.arange(2, self.args.top_k + 2).float().to(predicted_relevance.device))

        # Compute ideal DCG (IDCG) by sorting relevance ideally (highest scores first)
        ideal_relevance, _ = torch.sort(predicted_relevance, dim=-1, descending=True)
        ideal_gains = ideal_relevance
        idcg_pred = (ideal_gains / discounts).sum(dim=-1)
        
        return idcg_pred
    

    def threshold(self, predicted_relevance):
        predicted_relevance = (predicted_relevance[:, :10].T > self.args.threshold).int().T

        return predicted_relevance


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerForQPP(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=1):
        super(TransformerForQPP, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )

    def forward(self, embed):
        # embeddings: [batch, top_k, 768]
        # positional_encoding = positional_encoding.cuda()
        # transformer_encoder = transformer_encoder.cuda()
        # embed = torch.rand((16, 10, 768), device='cuda') # example tensor [batch, top_k, d_model]

        # add positional encoding
        embed = self.positional_encoding(embed)  # [batch, top_k, 768]

        # Transformer Encoder
        embed = self.transformer_encoder(embed)  # [top_k, batch, 768]
        
        # No Pooling, use all 
        # embed = embed  # [batch, top_k, 768]
        
        return embed  # [batch, top_k, 768]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, mlp_bias=True):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=mlp_bias)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)

        return x
