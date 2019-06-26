import mxnet as mx
import math
import torch
import numpy as np
from torch.nn import Module, Parameter
import torch.nn.functional as F

def l2_norm(input, axis=1):

    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Arcface_mxnet_like(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5, kernel=None):

        super(Arcface_mxnet_like, self).__init__()
        self.classnum = classnum

        if kernel is None:
            self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        else:
            self.kernel = Parameter(kernel)

        self.margin = m  # the margin value, default is 0.5
        self.scale = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)

    def forward(self, embeddings, label):
        # weights norm

        embeddings_norm = F.normalize(embeddings) * self.scale
        kernel_norm = F.normalize(self.kernel, dim=0)

        cos_theta = torch.mm(embeddings_norm, kernel_norm)

        mx_zy = cos_theta.gather(1, label.view(-1, 1))

        mx_cos_t = mx_zy / self.scale

        mx_cos_m = math.cos(self.margin)
        mx_sin_m = math.sin(self.margin)

        mx_mm = math.sin(math.pi - self.margin)

        mx_threshold = math.cos(math.pi - self.margin)

        mx_cond_v = mx_cos_t - mx_threshold

        mx_body = mx_cos_t * mx_cos_t
        mx_body = 1.0 - mx_body
        mx_sin_t = torch.sqrt(mx_body)

        mx_new_zy = mx_cos_t * mx_cos_m

        mx_b = mx_sin_t * mx_sin_m

        mx_new_zy = mx_new_zy - mx_b
        mx_new_zy = mx_new_zy * self.scale

        mx_zy_keep = mx_zy - self.scale * mx_mm

        mx_cond_mask = mx_cond_v <= 0

        mx_new_zy[mx_cond_mask] = mx_zy_keep[mx_cond_mask]

        mx_diff = mx_new_zy - mx_zy

        answer = cos_theta + mx_diff

        return answer


class ArcMarginProduct_wujiyang(Module):
    # implementation from https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py

    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50,
                 easy_margin=False, kernel = None):

        super(ArcMarginProduct_wujiyang, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m

        if kernel is None:
            self.weight = Parameter(torch.Tensor(out_feature, in_feature))
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            self.weight = Parameter(kernel)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


def obtain_output_from_mxnet(num_clases, emb_size, n_elements, np_labels, np_embeddings, np_weights):

    np_weights = np_weights.transpose()

    gt_label = mx.symbol.Variable("labels", shape=(n_elements, 1))

    mx_embedding = mx.symbol.Variable("mx_embedding",  shape=(n_elements, emb_size))

    _weight = mx.symbol.Variable("fc7_weight", shape=(num_clases, emb_size))

    _weight = mx.symbol.L2Normalization(_weight, mode='instance')

    nembedding = mx.symbol.L2Normalization(data=mx_embedding, mode='instance', name='fc1n') * scale

    fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True,
                                num_hidden=num_clases, name='fc7')

    zy = mx.sym.pick(fc7, gt_label, axis=1)

    cos_t = zy/scale

    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    mm = math.sin(math.pi - margin)

    threshold = math.cos(math.pi - margin)

    cond_v = cos_t - threshold
    cond = mx.symbol.Activation(data=cond_v, act_type='relu')

    body = cos_t * cos_t
    body = 1.0 - body

    sin_t = mx.sym.sqrt(body)
    new_zy = cos_t * cos_m

    b = sin_t * sin_m

    new_zy = new_zy - b
    new_zy = new_zy * scale

    zy_keep = zy - scale * mm

    selected_zy = mx.sym.where(cond, new_zy, zy_keep)

    diff = selected_zy - zy
    diff = mx.sym.expand_dims(diff, 1)

    gt_one_hot = mx.sym.one_hot(gt_label, depth=num_clases, on_value=1.0, off_value=1.0)
    body = mx.sym.broadcast_mul(gt_one_hot, diff)

    fc7 = fc7 + body

    mx_embeddings = mx.nd.array(np_embeddings)
    mx_labels = mx.nd.array(np_labels)
    mx_weights = mx.nd.array(np_weights)

    output = fc7.bind(mx.cpu(), {'mx_embedding': mx_embeddings, 'fc7_weight': mx_weights, "labels": mx_labels})

    print("mxnet output : ", output.forward())


num_clases = 5
emb_size = 5
n_elements = 5

np_labels = np.array([0, 3, 1, 4, 2])

np_embeddings = np.random.rand(n_elements, emb_size)

np_weights = np.random.rand(emb_size, num_clases)

scale = 64.
margin = 0.5

obtain_output_from_mxnet(num_clases, emb_size, n_elements,
                         np_labels, np_embeddings, np_weights)

print("-" * 50)

manual_kernel = torch.Tensor(np_weights)
torch_labels = torch.tensor([0, 3, 1, 4, 2], dtype=torch.long)

torch_embeddings = torch.Tensor(np_embeddings)
model_head_william = Arcface_mxnet_like(embedding_size=5, classnum=num_clases, kernel=manual_kernel)
res_william = model_head_william.forward(torch_embeddings, torch_labels)

print("answer arcface pytorch mxnet like : ", res_william)

loss = torch.nn.CrossEntropyLoss()
perte_william = loss.forward(res_william.squeeze(), torch_labels)
print("perte william : ", perte_william)


model_head_wujiyan = ArcMarginProduct_wujiyang(out_feature=emb_size,in_feature=num_clases,  kernel=manual_kernel,
                                               s= 64)
res_wujiyang = model_head_wujiyan.forward(torch_embeddings, torch_labels)
perte_wujiyang = loss.forward(res_wujiyang, torch_labels)

print("response wujiyan : ", res_wujiyang)
