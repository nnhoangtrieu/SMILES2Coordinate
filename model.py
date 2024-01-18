import torch 
import torch.nn as nn
import torch.nn.functional as F

class NN_Attention(nn.Module): # Neural Network Attention 
    def __init__(self, dim_model):
        super(NN_Attention, self).__init__()
        self.Wa = nn.Linear(dim_model, dim_model)
        self.Ua = nn.Linear(dim_model, dim_model)
        self.Va = nn.Linear(dim_model, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)

        context = torch.bmm(weights, keys)

        return context, weights 
    

class DP_Attention(nn.Module) : # Dot Product Attention
    def __init__(self, dim_model, num_head) :
        super(DP_Attention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.dim_head = dim_model // num_head

        self.Q = nn.Linear(dim_model, dim_model)
        self.K = nn.Linear(dim_model, dim_model)
        self.V = nn.Linear(dim_model, dim_model)

        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, Q, K, V) :
        B = Q.size(0) # Shape Q, K, V: (B, longest_smi, dim_model)

        Q, K, V = self.Q(Q), self.K(K), self.V(V)

        len_Q, len_K, len_V = Q.size(1), K.size(1), V.size(1)

        Q = Q.reshape(B, self.num_head, len_Q, self.dim_head)
        K = K.reshape(B, self.num_head, len_K, self.dim_head)
        V = V.reshape(B, self.num_head, len_V, self.dim_head)
        
        K_T = K.transpose(2,3).contiguous()

        attn_score = Q @ K_T

        attn_score = attn_score / (self.dim_head ** 1/2)

        attn_distribution = torch.softmax(attn_score, dim = -1)

        attn = attn_distribution @ V

        attn = attn.reshape(B, len_Q, self.num_head * self.dim_head)
        
        attn = self.out(attn)

        return attn, attn_distribution
    


class Encoder(nn.Module) : 
    def  __init__(self, dim_model, num_head, num_layer, dropout, atom_dic) :
        super(Encoder, self).__init__()
        self.Embedding = nn.Embedding(len(atom_dic), dim_model)
        self.Dropout = nn.Dropout(dropout)
        self.Encoder_Blocks = nn.ModuleList([Encoder_Block(dim_model, num_head, dropout) for _ in range(num_layer)])


    def forward(self, x) :
        x = self.Dropout(self.Embedding(x))

        for block in self.Encoder_Blocks : 
            x = block(x)
        
        return x 
    

class Encoder_Block(nn.Module) :
    def __init__(self, dim_model, num_head, dropout) : 
        super(Encoder_Block, self).__init__() 
        self.Self_Attention = DP_Attention(dim_model, num_head)
        self.LSTM = nn.LSTM(dim_model, dim_model, batch_first = True) 
        self.LayerNorm = nn.LayerNorm(dim_model)

    def forward(self, x) :  
        attn, self_attn = self.Self_Attention(x, x, x)

        x = self.LayerNorm(x + attn)

        all_state, (h, c) = self.LSTM(x) 

        return all_state, self_attn
    