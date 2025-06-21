import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size, n_node, n_price, img_emb_size, text_emb_size):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.n_node = n_node
        self.n_price = n_price
        self.img_emb_size = img_emb_size
        self.text_emb_size = text_emb_size

        self.dif2one_mlp = nn.Linear(3 * self.emb_size, self.emb_size)

        self.w_pv = nn.Linear(self.emb_size, self.emb_size)

        self.w_vp = nn.Linear(self.emb_size, self.emb_size)

        self.tran_pv = nn.Linear(self.emb_size, self.emb_size)
        self.tran_pc = nn.Linear(self.emb_size, self.emb_size)

        self.mat_pv = nn.Parameter(torch.Tensor(self.n_price, 1))

        self.a_i_g = nn.Linear(self.emb_size, self.emb_size)
        self.b_i_g = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_i = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gi1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gi3 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_p = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gp1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gp3 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_c = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gc1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gc3 = nn.Linear(self.emb_size, self.emb_size)

        self.a_o_g_b = nn.Linear(self.emb_size * 3, self.emb_size)
        self.b_o_gb1 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gb2 = nn.Linear(self.emb_size, self.emb_size)
        self.b_o_gb3 = nn.Linear(self.emb_size, self.emb_size)

        self.dropout10 = nn.Dropout(0.1)
        self.dropout20 = nn.Dropout(0.2)
        self.dropout30 = nn.Dropout(0.3)
        self.dropout40 = nn.Dropout(0.4)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout60 = nn.Dropout(0.6)
        self.dropout70 = nn.Dropout(0.7)

    def forward(self, adjacency, adjacency_pv, adjacency_vp, embedding, pri_emb, img_emb, text_emb):
        # updating embeddings with different types
        # convert image_emb and text_emb to the embeddings dimension as item_emb
        image_embeddings = self.img_mlp(img_emb)
        text_embeddings = self.text_mlp(text_emb)
        price_embeddings = self.pri_mlp(pri_emb)
        id_embeddings = self.id_mlp(embedding)


        return id_embeddings, image_embeddings, text_embeddings, price_embeddings


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=2, dropout=0, initializer_range=0.02):
        super(MultiHeadSelfAttention, self).__init__()
        self.config = list()

        self.hidden_size = hidden_size
        self.head_num = head_num
        if (self.hidden_size) % head_num != 0:
            raise ValueError(self.head_num, "error")
        self.head_dim = self.hidden_size // self.head_num

        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.key = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.value = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.concat_weight = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        torch.nn.init.normal_(self.query.weight, 0, initializer_range)
        torch.nn.init.normal_(self.key.weight, 0, initializer_range)
        torch.nn.init.normal_(self.value.weight, 0, initializer_range)
        torch.nn.init.normal_(self.concat_weight.weight, 0, initializer_range)
        self.dropout = torch.nn.Dropout(dropout)

    def dot_score(self, encoder_output):
        query = self.dropout(self.query(encoder_output))
        key = self.dropout(self.key(encoder_output))
        # head_num * batch_size * session_length * head_dim
        querys = torch.stack(query.chunk(self.head_num, -1), 0)
        keys = torch.stack(key.chunk(self.head_num, -1), 0)
        # head_num * batch_size * session_length * session_length
        dots = querys.matmul(keys.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        #         print(len(dots),dots[0].shape)
        return dots

    def forward(self, encoder_outputs, mask=None):
        attention_energies = self.dot_score(encoder_outputs)
        value = self.dropout(self.value(encoder_outputs))

        values = torch.stack(value.chunk(self.head_num, -1))

        if mask is not None:
            eye = torch.eye(mask.shape[-1]).to('cuda')
            new_mask = torch.clamp_max((1 - (1 - mask.float()).unsqueeze(1).permute(0, 2, 1).bmm(
                (1 - mask.float()).unsqueeze(1))) + eye, 1)
            attention_energies = attention_energies - new_mask * 1e12
            weights = F.softmax(attention_energies, dim=-1)
            weights = weights * (1 - new_mask)
        else:
            weights = F.softmax(attention_energies, dim=2)

        # head_num * batch_size * session_length * head_dim
        outputs = weights.matmul(values)
        # batch_size * session_length * hidden_size
        outputs = torch.cat([outputs[i] for i in range(outputs.shape[0])], dim=-1)
        outputs = self.dropout(self.concat_weight(outputs))

        return outputs


class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, initializer_range=0.02):
        super(PositionWiseFeedForward, self).__init__()
        self.final1 = torch.nn.Linear(hidden_size, hidden_size * 4, bias=True)
        self.final2 = torch.nn.Linear(hidden_size * 4, hidden_size, bias=True)
        torch.nn.init.normal_(self.final1.weight, 0, initializer_range)
        torch.nn.init.normal_(self.final2.weight, 0, initializer_range)

    def forward(self, x):
        x = F.relu(self.final1(x))
        x = self.final2(x)
        return x


class TransformerLayer(torch.nn.Module):
    def __init__(self, hidden_size, activate="relu", head_num=4, dropout=0, attention_dropout=0,
                 initializer_range=0.02):
        super(TransformerLayer, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.mh = MultiHeadSelfAttention(hidden_size=hidden_size, activate=activate, head_num=head_num,
                                         dropout=attention_dropout, initializer_range=initializer_range)
        self.pffn = PositionWiseFeedForward(hidden_size, initializer_range=initializer_range)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, encoder_outputs, mask=None):
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.mh(encoder_outputs, mask)))
        encoder_outputs = self.layer_norm(encoder_outputs + self.dropout(self.pffn(encoder_outputs)))
        return encoder_outputs

class MLP_merger(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_merger, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, out_size)
        self.mlp_2 = nn.Linear(out_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        return emb_trans

class MLP_conspace(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_conspace, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        return emb_trans

class MLP_adp(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_adp, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.Tanh()
        self.mlp_1 = nn.Linear(input_size, int(input_size / 4))
        self.mlp_2 = nn.Linear(int(input_size / 4), int(input_size / 8))
        self.mlp_3 = nn.Linear(int(input_size / 8), out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_3(emb_trans)))
        return emb_trans

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SemanticContrast(Module):
    def __init__(self, n_node, lr, layers, l2, alpha, beta, dataset, num_heads=4, emb_size=64, batch_size=100, num_item_negatives=100,num_sess_negatives=50):
        super(SemanticContrast, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.alpha = alpha
        self.beta = beta
        self.num_item_negatives = num_item_negatives
        self.num_sess_negatives = num_sess_negatives

        self.id_emb = nn.Embedding(self.n_node, self.emb_size)
        self.meger_id_img_txt = MLP_merger(self.emb_size*3, self.emb_size, dropout=0.5)
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)

        self.gate_w11 = nn.Linear(self.emb_size, self.emb_size)
        self.gate_w21 = nn.Linear(self.emb_size, self.emb_size)
        self.gate_w12 = nn.Linear(self.emb_size, self.emb_size)
        self.gate_w22 = nn.Linear(self.emb_size, self.emb_size)

        self.gate_w3 = nn.Linear(self.emb_size, self.emb_size)

        self.adp_item_pos_w = nn.Linear(self.emb_size*2, self.emb_size*2)
        self.adp_item_neg_w = nn.Linear(self.emb_size * 2, self.emb_size*2)
        self.adp_item_pos_mu = nn.Linear(self.emb_size*2, 1)
        self.adp_item_neg_mu = nn.Linear(self.emb_size*2, 1)

        self.adp_sess_pos_w = nn.Linear(self.emb_size * 2, self.emb_size*2)
        self.adp_sess_neg_w = nn.Linear(self.emb_size * 2, self.emb_size*2)
        self.adp_sess_pos_mu = nn.Linear(self.emb_size*2, 1)
        self.adp_sess_neg_mu = nn.Linear(self.emb_size*2, 1)


        # introducing text&image embeddings
        emb_path_pre = './datasets/'+ dataset + '/pca/'

        emb_path = emb_path_pre + 'textMatrixpca.npy'
        weights = np.array(np.load(emb_path))
        self.text_emb_ori = nn.Embedding(self.n_node, self.emb_size)
        self.text_emb_ori.weight.data.copy_(torch.from_numpy(weights))

        emb_path = emb_path_pre + 'textMatrixSwapwordpca.npy'
        weights = np.array(np.load(emb_path))
        self.text_emb_Swapword = nn.Embedding(self.n_node, self.emb_size)
        self.text_emb_Swapword.weight.data.copy_(torch.from_numpy(weights))


        emb_path = emb_path_pre + 'textMatrixBertsubpca.npy'
        weights = np.array(np.load(emb_path))
        self.text_emb_Bertsub = nn.Embedding(self.n_node, self.emb_size)
        self.text_emb_Bertsub.weight.data.copy_(torch.from_numpy(weights))

        emb_path = emb_path_pre + 'imgMatrixpca.npy'
        weights = np.array(np.load(emb_path))
        self.img_emb_ori = nn.Embedding(self.n_node, self.emb_size)
        self.img_emb_ori.weight.data.copy_(torch.from_numpy(weights))

        emb_path = emb_path_pre + 'imgMatrixGaussianNoisepca.npy'
        weights = np.array(np.load(emb_path))
        self.img_emb_GaussianNoise = nn.Embedding(self.n_node, self.emb_size)
        self.img_emb_GaussianNoise.weight.data.copy_(torch.from_numpy(weights))

        emb_path = emb_path_pre + 'imgMatrixCroppca.npy'
        weights = np.array(np.load(emb_path))
        self.img_emb_Crop = nn.Embedding(self.n_node, self.emb_size)
        self.img_emb_Crop.weight.data.copy_(torch.from_numpy(weights))

        # constrat MLP to same space MLP_conspace
        self.sess_con_mlp = torch.nn.ModuleList([MLP_conspace(self.emb_size, self.emb_size, dropout=0.5) for _ in range(7)])
        self.item_con_mlp = torch.nn.ModuleList(
            [MLP_conspace(self.emb_size, self.emb_size, dropout=0.5) for _ in range(7)])
        self.mlp_adaptive_loss = torch.nn.ModuleList([MLP_adp(self.emb_size*3, 1, dropout=0.5) for _ in range(2)])
        self.active = nn.ReLU()

        # self_attention
        if self.emb_size % num_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (emb_size, num_heads))
            # 参数定义
        self.num_heads = num_heads  # 4
        self.attention_head_size = int(self.emb_size / self.num_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_heads * self.attention_head_size)
        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(self.emb_size, self.emb_size)  # 128, 128
        self.key = nn.Linear(self.emb_size, self.emb_size)
        self.value = nn.Linear(self.emb_size, self.emb_size)


        # gate5 & gate6


        self.dropout = nn.Dropout(0.5)
        self.emb_dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout5 = nn.Dropout(0.5)
        self.dropout7 = nn.Dropout(0.7)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_emb_final, img_emb_ori, img_emb_GaussianNoise, img_emb_Crop, text_emb_ori, text_emb_Swapword, text_emb_Bertsub, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        mask = mask.float().unsqueeze(-1)

        id_embedding = torch.cat([zeros, item_emb_final], 0)
        img_ori_embedding = torch.cat([zeros, img_emb_ori], 0)
        img_GaussianNoise_embedding = torch.cat([zeros, img_emb_GaussianNoise], 0)
        img_Crop_embedding = torch.cat([zeros, img_emb_Crop], 0)

        text_ori_embedding = torch.cat([zeros, text_emb_ori], 0)
        text_Swapword_embedding = torch.cat([zeros, text_emb_Swapword], 0)
        text_Bertsub_embedding = torch.cat([zeros, text_emb_Bertsub], 0)

        # get = lambda i: item_embedding[reversed_sess_item[i]]
        # seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)

        id_seq_h = self.get_seq_element_emb(id_embedding, session_item)

        augimg1_seq_h = self.get_seq_element_emb(img_ori_embedding, session_item)
        augimg5_seq_h = self.get_seq_element_emb(img_GaussianNoise_embedding, session_item)
        augimg6_seq_h = self.get_seq_element_emb(img_Crop_embedding, session_item)

        augtxt1_seq_h = self.get_seq_element_emb(text_ori_embedding, session_item)
        augtxt2_seq_h = self.get_seq_element_emb(text_Swapword_embedding, session_item)
        augtxt5_seq_h = self.get_seq_element_emb(text_Bertsub_embedding, session_item)

        id_seq_emb = self.seq_encoder(id_seq_h, session_item, mask)
        augimg1_seq_emb = self.seq_encoder(augimg1_seq_h, session_item, mask)
        augimg2_seq_emb = self.seq_encoder(augimg5_seq_h, session_item, mask)
        augimg3_seq_emb = self.seq_encoder(augimg6_seq_h, session_item, mask)

        augtxt1_seq_emb = self.seq_encoder(augtxt1_seq_h, session_item, mask)
        augtxt2_seq_emb = self.seq_encoder(augtxt2_seq_h, session_item, mask)
        augtxt3_seq_emb = self.seq_encoder(augtxt5_seq_h, session_item, mask)



        return id_seq_emb, augimg1_seq_emb, augimg2_seq_emb,augimg3_seq_emb,augtxt1_seq_emb, augtxt2_seq_emb, augtxt3_seq_emb

    def fusion_img_text_gate(self, id_emb, img_emb_ori, text_emb_ori):
        id_embedding = id_emb
        merge_temp = torch.cat([id_embedding, img_emb_ori,text_emb_ori], 1)
        temp_emb = self.meger_id_img_txt(merge_temp)
        g1= torch.tanh(self.gate_w11(temp_emb) + self.gate_w12(img_emb_ori))
        g2 = torch.tanh(self.gate_w21(temp_emb) + self.gate_w22(text_emb_ori))

        item_final_emb = id_embedding + g1 * img_emb_ori + g2 * text_emb_ori


        return item_final_emb

    def fusion_img_text(self, id_emb, img_emb_ori, text_emb_ori):
        id_embedding = id_emb
        merge_temp = torch.cat([id_embedding, img_emb_ori,text_emb_ori], 1)
        item_emb_meger = self.meger_id_img_txt(merge_temp)
        return item_emb_meger

    def item_contrastive(self, item_emb_final, img_emb_ori, img_emb_GaussianNoise, img_emb_Crop, text_emb_ori, text_emb_Swapword, text_emb_Bertsub):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)

        img_emb_ori_table = torch.cat([zeros, img_emb_ori], 0)
        img_emb_GaussianNoise_table = torch.cat([zeros, img_emb_GaussianNoise], 0)
        img_emb_Crop_table = torch.cat([zeros, img_emb_Crop], 0)

        text_emb_ori_table = torch.cat([zeros, text_emb_ori], 0)
        text_Swapword_table = torch.cat([zeros, text_emb_Swapword], 0)
        text_Bertsub_table = torch.cat([zeros, text_emb_Bertsub], 0)

        img_emb_ori_mlp = self.item_con_mlp[1](img_emb_ori)
        img_emb_GaussianNoise_mlp = self.item_con_mlp[2](img_emb_GaussianNoise)
        img_emb_Crop_mlp = self.item_con_mlp[3](img_emb_Crop)

        text_emb_ori_mlp = self.item_con_mlp[4](text_emb_ori)
        text_emb_Swapword_mlp = self.item_con_mlp[5](text_emb_Swapword)
        text_emb_Bertsub_mlp = self.item_con_mlp[6](text_emb_Bertsub)


        pos_emb = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0)
        # pos_no_emb = torch.cuda.FloatTensor(self.n_node, self.emb_size).fill_(0)

        select_view = torch.randint(6, (self.n_node,1))
        pos_view_matrix = trans_to_cuda(select_view.expand_as(pos_emb))

        pos_select_mat = torch.where(pos_view_matrix == 0, torch.tensor([1.], device=self.device),
                           torch.tensor([0.], device=self.device))
        pos_emb = pos_emb + pos_select_mat * img_emb_ori_mlp
        # pos_no_emb = pos_no_emb + pos_select_mat * img_emb_Hflip

        pos_select_mat = torch.where(pos_view_matrix == 1, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_emb = pos_emb + pos_select_mat * img_emb_GaussianNoise_mlp

        pos_select_mat = torch.where(pos_view_matrix == 2, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_emb = pos_emb + pos_select_mat * img_emb_Crop_mlp

        pos_select_mat = torch.where(pos_view_matrix == 3, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_emb = pos_emb + pos_select_mat * text_emb_ori_mlp

        pos_select_mat = torch.where(pos_view_matrix == 4, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_emb = pos_emb + pos_select_mat * text_emb_Swapword_mlp

        pos_select_mat = torch.where(pos_view_matrix == 5, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_emb = pos_emb + pos_select_mat * text_emb_Bertsub_mlp
        # pos_no_emb = pos_no_emb + pos_select_mat * text_emb_Bertsub


        neg_emb = torch.cuda.FloatTensor(self.n_node, self.num_item_negatives, self.emb_size).fill_(0)
        # neg_no_emb = torch.cuda.FloatTensor(self.n_node, self.num_item_negatives, self.emb_size).fill_(0)

        neg_index = torch.randint(self.n_node, (self.n_node, self.num_item_negatives))
        neg_aug0_no_emb = img_emb_ori_table[neg_index]
        neg_aug1_no_emb = img_emb_GaussianNoise_table[neg_index]
        neg_aug2_no_emb = img_emb_Crop_table[neg_index]

        neg_aug3_no_emb = text_emb_ori_table[neg_index]
        neg_aug4_no_emb = text_Swapword_table[neg_index]
        neg_aug5_no_emb = text_Bertsub_table[neg_index]

        neg_aug0_emb = self.item_con_mlp[1](neg_aug0_no_emb)
        neg_aug1_emb = self.item_con_mlp[2](neg_aug1_no_emb)
        neg_aug2_emb = self.item_con_mlp[3](neg_aug2_no_emb)
        neg_aug3_emb = self.item_con_mlp[4](neg_aug3_no_emb)
        neg_aug4_emb = self.item_con_mlp[5](neg_aug4_no_emb)
        neg_aug5_emb = self.item_con_mlp[6](neg_aug5_no_emb)


        neg_view_matrix = pos_view_matrix.unsqueeze(1).expand_as(neg_emb)

        neg_select_mat = torch.where(neg_view_matrix == 0, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_emb = neg_emb + neg_select_mat * neg_aug0_emb
        # neg_no_emb = neg_no_emb + neg_select_mat * neg_aug0_no_emb

        neg_select_mat = torch.where(neg_view_matrix == 1, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_emb = neg_emb + neg_select_mat * neg_aug1_emb
        # neg_no_emb = neg_no_emb + neg_select_mat * neg_aug1_no_emb

        neg_select_mat = torch.where(neg_view_matrix == 2, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_emb = neg_emb + neg_select_mat * neg_aug2_emb
        # neg_no_emb = neg_no_emb + neg_select_mat * neg_aug2_no_emb

        neg_select_mat = torch.where(neg_view_matrix == 3, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_emb = neg_emb + neg_select_mat * neg_aug3_emb

        neg_select_mat = torch.where(neg_view_matrix == 4, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_emb = neg_emb + neg_select_mat * neg_aug4_emb

        neg_select_mat = torch.where(neg_view_matrix == 5, torch.tensor([1.], device=self.device),
                                 torch.tensor([0.], device=self.device))
        neg_emb = neg_emb + neg_select_mat * neg_aug5_emb

        item_emb_mlp = self.item_con_mlp[0](item_emb_final)

        # con_loss = self.item_contrast_loss_random(item_emb_mlp, pos_emb, neg_emb, item_emb_final, pos_no_emb, neg_no_emb)
        con_loss = self.item_contrast_loss_random(item_emb_mlp, pos_emb, neg_emb, item_emb_mlp, pos_emb, neg_emb)
        return con_loss

    def sess_contrastive(self, sess_emb, aug1_sess_emb, aug2_sess_emb, aug3_sess_emb, aug4_sess_emb, aug5_sess_emb, aug6_sess_emb):
        tau = 1
        sess_emb_mlp = self.sess_con_mlp[0](sess_emb)
        aug_sess_emb_c1 = self.sess_con_mlp[1](aug1_sess_emb)
        aug_sess_emb_c2 = self.sess_con_mlp[2](aug2_sess_emb)
        aug_sess_emb_c3 = self.sess_con_mlp[3](aug3_sess_emb)
        aug_sess_emb_c4 = self.sess_con_mlp[4](aug4_sess_emb)
        aug_sess_emb_c5 = self.sess_con_mlp[5](aug5_sess_emb)
        aug_sess_emb_c6 = self.sess_con_mlp[6](aug6_sess_emb)

        pos_sess_emb = torch.cuda.FloatTensor(self.batch_size, self.emb_size).fill_(0)
        select_view = torch.randint(6, (self.batch_size, 1))
        pos_view_matrix = trans_to_cuda(select_view.expand_as(pos_sess_emb))

        # pos_sess_no_emb = torch.cuda.FloatTensor(self.batch_size, self.emb_size).fill_(0)

        pos_select_mat = torch.where(pos_view_matrix == 0, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_sess_emb = pos_sess_emb + pos_select_mat * aug_sess_emb_c1
        # pos_sess_no_emb = pos_sess_no_emb + pos_select_mat * aug1_sess_emb

        pos_select_mat = torch.where(pos_view_matrix == 1, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_sess_emb = pos_sess_emb + pos_select_mat * aug_sess_emb_c2
        # pos_sess_no_emb = pos_sess_no_emb + pos_select_mat * aug2_sess_emb

        pos_select_mat = torch.where(pos_view_matrix == 2, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_sess_emb = pos_sess_emb + pos_select_mat * aug_sess_emb_c3
        # pos_sess_no_emb = pos_sess_no_emb + pos_select_mat * aug3_sess_emb

        pos_select_mat = torch.where(pos_view_matrix == 3, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_sess_emb = pos_sess_emb + pos_select_mat * aug_sess_emb_c4

        pos_select_mat = torch.where(pos_view_matrix == 4, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_sess_emb = pos_sess_emb + pos_select_mat * aug_sess_emb_c5

        pos_select_mat = torch.where(pos_view_matrix == 5, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        pos_sess_emb = pos_sess_emb + pos_select_mat * aug_sess_emb_c6

        neg_sess_emb = torch.cuda.FloatTensor(self.batch_size, self.num_sess_negatives, self.emb_size).fill_(0)

        # neg_sess_no_emb = torch.cuda.FloatTensor(self.batch_size, self.num_sess_negatives, self.emb_size).fill_(0)

        neg_aug1_emb = aug_sess_emb_c1.unsqueeze(0).expand_as(neg_sess_emb)
        neg_aug2_emb = aug_sess_emb_c2.unsqueeze(0).expand_as(neg_sess_emb)
        neg_aug3_emb = aug_sess_emb_c3.unsqueeze(0).expand_as(neg_sess_emb)
        neg_aug4_emb = aug_sess_emb_c4.unsqueeze(0).expand_as(neg_sess_emb)
        neg_aug5_emb = aug_sess_emb_c5.unsqueeze(0).expand_as(neg_sess_emb)
        neg_aug6_emb = aug_sess_emb_c6.unsqueeze(0).expand_as(neg_sess_emb)
        #
        neg_view_matrix = pos_view_matrix.unsqueeze(1).expand_as(neg_sess_emb)
        neg_select_mat = torch.where(neg_view_matrix == 0, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_sess_emb = neg_sess_emb + neg_select_mat*neg_aug1_emb

        neg_select_mat = torch.where(neg_view_matrix == 1, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_sess_emb = neg_sess_emb + neg_select_mat * neg_aug2_emb

        neg_select_mat = torch.where(neg_view_matrix == 2, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_sess_emb = neg_sess_emb + neg_select_mat * neg_aug3_emb

        neg_select_mat = torch.where(neg_view_matrix == 3, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_sess_emb = neg_sess_emb + neg_select_mat * neg_aug4_emb

        neg_select_mat = torch.where(neg_view_matrix == 4, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_sess_emb = neg_sess_emb + neg_select_mat * neg_aug5_emb

        neg_select_mat = torch.where(neg_view_matrix == 5, torch.tensor([1.], device=self.device),
                                     torch.tensor([0.], device=self.device))
        neg_sess_emb = neg_sess_emb + neg_select_mat * neg_aug6_emb
        # con_loss = self.sess_contrast_loss_random(sess_emb_mlp, pos_sess_emb, neg_sess_emb, sess_emb, pos_sess_no_emb, neg_sess_no_emb)
        con_loss = self.sess_contrast_loss_random(sess_emb_mlp, pos_sess_emb, neg_sess_emb, sess_emb_mlp, pos_sess_emb, neg_sess_emb)
        return con_loss

    def transpose_for_scores(self, x, attention_head_size):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_heads, attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)

    def get_seq_element_emb(self, emb_table, session_item):
        get = lambda i: emb_table[session_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(session_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        return seq_h
    def seq_encoder(self, seq_h, session_item, mask):
        # self-attention to get price preference
        attention_mask = mask.permute(0, 2, 1).unsqueeze(1)  # [bs, 1, 1, seqlen] 增加维度
        attention_mask = (1.0 - attention_mask) * -10000.0

        mixed_query_layer = self.query(seq_h)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(seq_h)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(seq_h)  # [bs, seqlen, hid_size]

        attention_head_size = int(self.emb_size / self.num_heads)
        query_layer = self.transpose_for_scores(mixed_query_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, attention_head_size)  # [bs, 8, seqlen, 16]
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(attention_head_size)  # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        attention_scores = attention_scores + attention_mask
        # 加上mask，将padding所在的表示直接-10000
        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.emb_size,)  # [bs, seqlen, 128]
        sa_result = context_layer.view(*new_context_layer_shape)
        item_pos = torch.tensor(range(1, seq_h.size()[1] + 1), device='cuda')
        item_pos = item_pos.unsqueeze(0).expand_as(session_item)

        item_pos = item_pos.type(torch.cuda.FloatTensor) * mask.squeeze(2)
        item_last_num = torch.max(item_pos, 1)[0].unsqueeze(1).expand_as(item_pos)
        last_pos_t = torch.where(item_pos - item_last_num >= 0, torch.tensor([1.0], device='cuda'),
                                 torch.tensor([0.0], device='cuda'))
        last_interest = last_pos_t.unsqueeze(2).expand_as(sa_result) * sa_result
        seq_emb = torch.sum(last_interest, 1)
        return seq_emb

    def item_contrast_loss(self, emb1, emb2, emb_table):
        tau = 1
        # img_textimg_mat = torch.matmul(emb1, emb2.permute(1, 0))
        # # softmax
        # # img_textimg_mat = torch.nn.Softmax(dim=0)(img_textimg_mat)
        # img_textimg_mat = img_textimg_mat / tau
        # img_textimg_mat = torch.exp(img_textimg_mat, out=None)
        # topk_img_values, topk_indice = torch.topk(img_textimg_mat, k=num_neg, dim=1)
        # # topk_values = nn.Softmax(dim=-1)(topk_values)
        # img_loss_one = torch.sum(torch.log10(torch.diag(img_textimg_mat)))
        # img_loss_two = torch.sum(torch.log10(torch.sum(topk_img_values, 1)))
        # # img_loss_three = torch.sum(torch.log10(torch.sum(topk_img_img_values, 1)))
        # loss = img_loss_two - img_loss_one
        con_sim_up =  torch.sum(emb1*emb2, 1)
        con_sim_down = torch.norm(emb1,2,1)* torch.norm(emb2,2,1) + 1e-12
        con_sim = con_sim_up/con_sim_down
        pos_score = torch.log10(torch.exp(con_sim/tau, out=None))
        pos_score = torch.sum(pos_score, 0)

        neg_index = torch.randint(self.n_node,(self.n_node, self.num_item_negatives))
        neg_emb = emb_table[neg_index]
        emb1_expand = emb1.unsqueeze(1).expand_as(neg_emb)
        con_sim_up = torch.sum(neg_emb * emb1_expand, 2)
        con_sim_down = torch.norm(emb1_expand, 2, 2) * torch.norm(neg_emb, 2, 2) + 1e-12
        neg_score = con_sim_up/con_sim_down
        neg_score = torch.exp(neg_score/tau, out=None)
        neg_score = torch.sum(neg_score, 1)
        neg_score = torch.sum(torch.log10(neg_score))
        loss = neg_score- pos_score

        return loss

    def item_contrast_loss_random(self, item_emb, pos_emb, neg_emb, item_no_emb, pos_no_emb, neg_no_emb):
        tau = 1
        # auto-adaptive contrastive loss
        adp_ori_emb = item_no_emb
        adp_pos_emb = pos_no_emb
        adp_neg_emb = torch.sum(neg_no_emb, 1)/self.num_item_negatives
        # # pos adaptive from all positives
        # adp_std_pos = self.dropout(torch.cat([adp_ori_emb, adp_pos_emb], -1))
        # adp_pos_lamda = self.adp_item_pos_mu(self.adp_item_pos_w(adp_std_pos))
        # adp_pos_lamda = adp_pos_lamda.squeeze(-1)
        # adp_pos_lamda = torch.softmax(adp_pos_lamda, 0)
        # # neg adaptive in pos
        # # adp_ori_emb_expand = adp_ori_emb.unsqueeze(1).expand_as(adp_neg_emb)
        # adp_std_neg = self.dropout(torch.cat([adp_ori_emb, adp_neg_emb], -1))
        # adp_neg_lamda = self.adp_item_neg_mu(self.adp_item_neg_w(adp_std_neg))
        # adp_neg_lamda = adp_neg_lamda.squeeze(-1)
        # adp_neg_lamda = torch.softmax(adp_neg_lamda, 0)

        # adpative loss all to single
        adp_std = self.dropout(torch.cat([adp_ori_emb, adp_pos_emb, adp_neg_emb], -1))
        adp_lamda = self.mlp_adaptive_loss[0](adp_std)
        adp_lamda = adp_lamda.squeeze(-1)
        adp_lamda = torch.softmax(adp_lamda, 0)



        con_sim_up =  torch.sum(item_emb*pos_emb, 1)
        con_sim_down = torch.norm(item_emb,2,1)* torch.norm(pos_emb,2,1) + 1e-12
        con_sim = con_sim_up/con_sim_down
        # pos_score = torch.softmax(adp_pos_lamda+adp_neg_lamda, 0)*torch.log10(torch.exp(con_sim/tau, out=None))
        pos_score = adp_lamda*torch.log10(torch.exp(con_sim/tau, out=None))
        pos_score = torch.sum(pos_score, 0)

        emb1_expand = item_emb.unsqueeze(1).expand_as(neg_emb)
        con_sim_up = torch.sum(neg_emb * emb1_expand, 2)
        con_sim_down = torch.norm(emb1_expand, 2, 2) * torch.norm(neg_emb, 2, 2) + 1e-12
        neg_score = con_sim_up/con_sim_down
        neg_score = torch.exp(neg_score/tau, out=None)
        neg_score = torch.sum(neg_score, 1)
        neg_score = torch.sum(torch.log10(neg_score))
        loss = neg_score- pos_score

        return loss

    def sess_contrast_loss_random(self, sess_emb, pos_emb, neg_emb, sess_no_emb, pos_no_emb, neg_no_emb):
        tau = 1
        # auto-adaptive contrastive loss
        adp_ori_emb = sess_no_emb
        adp_pos_emb = pos_no_emb
        adp_neg_emb = torch.sum(neg_no_emb, 1)/self.num_sess_negatives
        # # pos adaptive from all positives
        # adp_std_pos = self.dropout(torch.cat([adp_ori_emb, adp_pos_emb], -1))
        # adp_pos_lamda = self.adp_sess_pos_mu(self.adp_sess_pos_w(adp_std_pos))
        # adp_pos_lamda = adp_pos_lamda.squeeze(-1)
        # adp_pos_lamda = torch.softmax(adp_pos_lamda, 0)
        # # neg adaptive in pos
        # # adp_ori_emb_expand = adp_ori_emb.unsqueeze(1).expand_as(adp_neg_emb)
        # adp_std_neg = self.dropout(torch.cat([adp_ori_emb, adp_neg_emb], -1))
        # adp_neg_lamda = self.adp_sess_neg_mu(self.adp_sess_neg_w(adp_std_neg))
        # adp_neg_lamda = adp_neg_lamda.squeeze(-1)
        # adp_neg_lamda = torch.softmax(adp_neg_lamda, 0)

        adp_std = self.dropout(torch.cat([adp_ori_emb, adp_pos_emb, adp_neg_emb], -1))
        adp_lamda = self.mlp_adaptive_loss[1](adp_std)
        adp_lamda = adp_lamda.squeeze(-1)
        adp_lamda = torch.softmax(adp_lamda, 0)


        con_sim_up = torch.sum(sess_emb * pos_emb, 1)
        con_sim_down = torch.norm(sess_emb, 2, 1) * torch.norm(pos_emb, 2, 1) + 1e-12
        con_sim = con_sim_up / con_sim_down
        # pos_score = torch.softmax(adp_pos_lamda+adp_neg_lamda, 0)*torch.log10(torch.exp(con_sim / tau, out=None))
        pos_score = adp_lamda * torch.log10(torch.exp(con_sim / tau, out=None))
        pos_score = torch.sum(pos_score, 0)

        emb1_expand = sess_emb.unsqueeze(1).expand_as(neg_emb)
        con_sim_up = torch.sum(neg_emb * emb1_expand, 2)
        con_sim_down = torch.norm(emb1_expand, 2, 2) * torch.norm(neg_emb, 2, 2) + 1e-12
        neg_score = con_sim_up / con_sim_down
        neg_score = torch.exp(neg_score / tau, out=None)
        neg_score = torch.sum(neg_score, 1)
        neg_score = torch.sum(torch.log10(neg_score))
        loss = neg_score - pos_score
        return loss

    def adpative_loss(self, ori_emb, pos_emb, neg_emb, w1):
        return None

    def forward(self, session_item, session_len, reversed_sess_item, mask):
        # session_item 是一个batch里的所有session [[23,34,0,0],[1,3,4,0]]

        id_emb = self.id_emb.weight
        img_emb_ori = self.img_emb_ori.weight
        img_emb_GaussianNoise = self.img_emb_GaussianNoise.weight
        img_emb_Crop = self.img_emb_Crop.weight


        text_emb_ori = self.text_emb_ori.weight
        text_emb_Swapword = self.text_emb_Swapword.weight
        text_emb_Bertsub = self.text_emb_Bertsub.weight

        # fusion item id&image&text
        item_emb_final = self.fusion_img_text_gate(id_emb, img_emb_ori, text_emb_ori)

        # item-level contrast
        item_con_loss = self.item_contrastive(item_emb_final, img_emb_ori, img_emb_GaussianNoise, img_emb_Crop, text_emb_ori, text_emb_Swapword, text_emb_Bertsub)

        # sequence encoder including item_ID&augmented sequence
        sess_emb, aug1_sess_emb, aug2_sess_emb, aug3_sess_emb, aug4_sess_emb,aug5_sess_emb, aug6_sess_emb= self.generate_sess_emb(item_emb_final, img_emb_ori, img_emb_GaussianNoise, img_emb_Crop, text_emb_ori, text_emb_Swapword, text_emb_Bertsub, session_item, session_len, reversed_sess_item, mask) #batch内session embeddings
        # session-level contrast
        sess_con_loss = self.sess_contrastive(sess_emb, aug1_sess_emb, aug2_sess_emb, aug3_sess_emb, aug4_sess_emb, aug5_sess_emb, aug6_sess_emb)
        con_loss = self.alpha * item_con_loss + self.beta * sess_con_loss
        return item_emb_final, sess_emb, con_loss


def perform(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i) # 得到一个batch里的数据
    # A_hat, D_hat = data.get_overlap(session_item)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    # A_hat = trans_to_cuda(torch.Tensor(A_hat))
    # D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_final, sess_emb, con_loss = model(session_item, session_len, reversed_sess_item, mask)
    scores = torch.mm(sess_emb, torch.transpose(item_emb_final, 1, 0))

    scores = trans_to_cuda(scores)
    return tar, scores, con_loss

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret

def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))

    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2


def sess_contrast_loss(emb1, emb2, num_neg=1000, tau=1):
    # consin_similarity
    sim_matrix_up = torch.matmul(emb1, emb2.permute(1, 0))
    sim_matrix_down = torch.matmul(torch.norm(emb1,2,1,keepdim=True), torch.norm(emb2,2,1,keepdim=True).permute(1, 0)) + 1e-12
    sim_matrix = sim_matrix_up/sim_matrix_down
    # softmax
    # img_textimg_mat = torch.nn.Softmax(dim=0)(img_textimg_mat)
    sim_matrix = sim_matrix / tau
    sim_matrix = torch.exp(sim_matrix, out=None)
    topk_values, topk_indice = torch.topk(sim_matrix, k=num_neg, dim=1)
    # topk_values = nn.Softmax(dim=-1)(topk_values)
    pos_score = torch.sum(torch.log10(torch.diag(sim_matrix)))
    neg_score = torch.sum(torch.log10(torch.sum(topk_values, 1)))
    loss = neg_score - pos_score
    return loss




def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size) #将session随机打乱，每x个一组（#session/batch_size)
    for i in slices:
        model.zero_grad()
        targets, scores, con_loss = perform(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        metrics['ndcg%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, con_loss = perform(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                    metrics['ndcg%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
                    metrics['ndcg%d' % K].append(1 / (np.log2(np.where(prediction == target)[0][0] + 2)))
    return metrics, total_loss


