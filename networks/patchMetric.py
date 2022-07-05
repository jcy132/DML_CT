from packaging import version
import torch
from torch import nn



class PatchMetricLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        '''
        :param feat_q: output feature
        :param feat_k: input feature
        :return:
        '''

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]

        # pos logit -> bmm (B*K, 1, dim) , (B*K, dim, 1)  -> inner product with same(B*K)dim
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minib  atch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch_k = torch.bmm(feat_k, feat_k.transpose(2, 1))              #(B, K, K)
        l_neg_curbatch_q = torch.bmm(feat_q, feat_q.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch_k.masked_fill_(diagonal, -10.0)  # (B, K, K )
        l_neg_curbatch_q.masked_fill_(diagonal, -10.0)  # (B, K, K )
        loss = -torch.exp(l_pos / self.opt.metric_T).mean() + torch.exp(l_neg_curbatch_k / self.opt.metric_T).mean() + torch.exp(l_neg_curbatch_q / self.opt.metric_T).mean()
        loss /= dim

        return loss
