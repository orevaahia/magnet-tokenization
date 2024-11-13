import torch
import torch.nn as nn
import torch.nn.functional as F

from shortening import downsample, upsample
from utils import compute_mean_with_padding


@torch.jit.script
def add_and_scale(tensor1, tensor2, alpha: float):
    return alpha * (tensor1 + tensor2)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm, activation_function):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            core_out = self.CoreNet(self.layer_norm(inp))
            output = core_out + inp
        else:
            core_out = self.CoreNet(inp)
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(
        self, n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
    ):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        del activation_function

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(self.d_model, 3 * n_head * d_head)
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), x.size(1), x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=3)

        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))

        x = x_padded.narrow(2, 1, x_padded.size(2) - 1).view_as(x)

        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask):
        # w is of size: T x B x C
        # r is of size: T x 1 x C
        # biases are of size: (n_head x d_head), we add the same bias to each token
        # attn_mask is of size (q_len x k_len)
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if self.pre_lnorm:
            w_head_q, w_head_k, w_head_v = self.qkv_net(self.layer_norm(w))
        else:
            w_heads = self.qkv_net(w)

        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)       # qlen x n_head x d_head

        # compute attention score
        rw_head_q = w_head_q + r_w_bias                                # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->bnij', rw_head_q, w_head_k)      # bsz x n_head x qlen x klen

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->bnij', rr_head_q, r_head_k)       # bsz x n_head x qlen x klen
        BD = self._rel_shift(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = add_and_scale(AC, BD, self.scale)

        # compute attention probability
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, None, :, :], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, None, :, :], -float('inf'))
        else:
            raise NotImplementedError

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,jbnd->ibnd', attn_prob, w_head_v)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        pre_lnorm,
        activation_function,
    ):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt, pre_lnorm, activation_function
        )
        self.pos_ff = PositionwiseFF(
            d_model,
            d_inner,
            dropout,
            pre_lnorm,
            activation_function,
        )

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None):
        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask)
        output = self.pos_ff(output)

        return output


class BoundaryPredictor(nn.Module):
    def __init__(self, d_model, d_inner, activation_function,
                 temp, prior, bp_type, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.bp_type = bp_type
        self.threshold = threshold

        if activation_function == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_function == 'gelu':
            activation_fn = torch.nn.GELU()

        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_inner),
            activation_fn,
            nn.Linear(d_inner, 1),
        )

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, hidden):
        # Hidden is of shape [seq_len x bs x d_model]
        # Boundaries we return are [bs x seq_len]

        boundary_logits = self.boundary_predictor(hidden).squeeze(-1).transpose(0, 1)
        boundary_probs = torch.sigmoid(boundary_logits)

        if self.bp_type == 'gumbel':
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=boundary_probs,
            )

            soft_boundaries = bernoulli.rsample()

            hard_boundaries = (soft_boundaries > self.threshold).float()
            hard_boundaries = (
                hard_boundaries - soft_boundaries.detach() + soft_boundaries
            )
        elif self.bp_type in ['entropy', 'unigram']:
            soft_boundaries = boundary_probs
            hard_boundaries = (soft_boundaries > self.threshold).float()

        return soft_boundaries, hard_boundaries

    def calc_loss_without_padding(self, preds, gt, attention_mask):
        """

        """
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())

        elif self.bp_type in ['gumbel']:
            assert attention_mask is not None and gt is None

            # create a mask based on attention_mask
            mask = attention_mask.eq(1)  # Mask is True where tokens are present, False for padding

            # apply the mask to predictions
            masked_preds = preds * mask.float()

            # Compute the sum of predictions for each example in the batch
            sum_preds = masked_preds.sum(dim=-1).unsqueeze(dim=-1)

            # Compute the total count of trials for each example in the batch
            total_count = mask.sum(dim=-1, keepdim=True).float()  # Number of non-padded tokens

            # compute the sum of predictions for each example in the batch
            # sum_preds = masked_preds.sum(dim=1)
            binomial = torch.distributions.binomial.Binomial(
                    total_count,
                    probs=torch.Tensor([self.prior]).to(preds.device)
                )
            loss_boundaries = -binomial.log_prob(
                    sum_preds
                ).mean()

            return loss_boundaries

    def calc_loss(self, preds, gt):
        # B x T
        if self.bp_type in ['entropy', 'unigram']:
            assert preds is not None and gt is not None
            return self.loss(preds, gt.float())
        elif self.bp_type in ['gumbel']:
            assert gt is None
            binomial = torch.distributions.binomial.Binomial(
                preds.size(-1),
                probs=torch.Tensor([self.prior]).to(preds.device)
            )
            loss_boundaries = -binomial.log_prob(
                preds.sum(dim=-1)
            ).mean() / preds.size(-1)

            return loss_boundaries

    def calc_stats(self, preds, gt):
        # B x T
        preds, gt = preds.bool(), gt.bool()
        TP = ((preds == gt) & preds).sum().item()
        FP = ((preds != gt) & preds).sum().item()
        FN = ((preds != gt) & (~preds)).sum().item()

        acc = (preds == gt).sum().item() / gt.numel()

        if TP == 0:
            precision, recall = 0, 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

        stats = {
            'acc': acc,
            'precision': precision,
            'recall': recall
        }

        return stats

class MagnetTransformerLM(nn.Module):
    def __init__(self, n_token, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, pre_lnorm, model_config,
                 activation_function, boundaries_type, spikes_left,
                 temp, all_script_ids_dict, script_to_id
                 ):
        super(MagnetTransformerLM, self).__init__()
        self.n_token = n_token
        self.script_to_id = script_to_id

        # when loading the pretrained config, the keys become strings instead of int, so we convert to int here
        are_all_script_keys_string = all(isinstance(value, str) for value in self.script_to_id.keys())
        if are_all_script_keys_string:
            self.script_to_id = {int(key): value for key, value in self.script_to_id.items() if key.isdigit()}

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
        self.drop = nn.Dropout(dropout)

        # Relative attention specific parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )
        self.r_r_bias = nn.Parameter(
            torch.Tensor(self.n_head, self.d_head).zero_()
        )

        assert pre_lnorm is False, "We didn't use pre_lnorm"

        def create_decoder_layers(n_layers):
            layers = nn.ModuleList([
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner, dropout,
                    dropatt=dropatt, pre_lnorm=pre_lnorm,
                    activation_function=activation_function)
                for _ in range(n_layers)
            ])

            return layers

        pre_layers, (shortened_layers, ), post_layers = eval(model_config)

        self.boundaries_type = boundaries_type
        self.is_bp = boundaries_type in ['unigram', 'entropy', 'gumbel']

        if post_layers == 0 and shortened_layers == 0:
            assert boundaries_type == 'none'
            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers)
            ])
        else:
            self.null_group = nn.Parameter(torch.Tensor(1, 1, d_model).zero_())
            nn.init.normal_(self.null_group)

            self.layers = nn.ModuleList([
                create_decoder_layers(pre_layers),
                create_decoder_layers(shortened_layers),
                create_decoder_layers(post_layers),
            ])

            self.down_ln = nn.LayerNorm(d_model)

            # Create boundary predictor layers
            if self.is_bp:
                self.script_to_bp_layers = nn.ModuleDict({script: BoundaryPredictor(
                    d_model=d_model,
                    d_inner=d_inner,
                    activation_function=activation_function,
                    temp=temp,
                    prior=pri,
                    bp_type=boundaries_type,
                )
                for i, (script, pri) in enumerate(zip(all_script_ids_dict.keys(), all_script_ids_dict.values()))
                })

                self.spikes_left = spikes_left

        self.final_cast = nn.Linear(d_model, n_token)
        #self.crit = torch.nn.CrossEntropyLoss(reduction='none', ignore_index= -100)
        self.crit = torch.nn.CrossEntropyLoss(ignore_index= -100)

    def _forward(self, core_input, layers):
        # Core_input is of size (T x B x C)
        qlen, _, _ = core_input.size()

        dec_attn_mask = torch.triu(
            core_input.new_ones(qlen, qlen), diagonal=1).bool()

        pos_seq = torch.arange(
            qlen - 1, -1, -1.0, device=core_input.device, dtype=core_input.dtype
        )

        pos_emb = self.pos_emb(pos_seq)
        pos_emb = self.drop(pos_emb)

        core_out = core_input
        for i, layer in enumerate(layers):
            core_out = layer(
                core_out, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask
            )

        return core_out

    def get_spikes(self, vector):
        total = torch.ones_like(vector).bool()

        for i in range(1, self.spikes_left + 1, 1):
            mask = vector[i:] > vector[:-i]
            total[i:] &= mask

        return total

    def compute_compression_rate(self, hard_boundaries, attention_mask):
        # Create a mask based on attention_mask
        mask = attention_mask.eq(1)  # Mask is True where tokens are present, False for padding

        # Apply the mask to hard_boundaries
        masked_hard_boundaries = hard_boundaries * mask.float()

        # Compute the total number of non-padded positions for each row in the batch
        num_non_padded_positions_per_row = mask.sum(dim=1).float()  # Count the number of non-padded positions for each row

        # Compute the sum of predictions only on non-padded positions for each row in the batch
        sum_hard_boundaries_non_padded_per_row = masked_hard_boundaries.sum(dim=1)  # Sum of hard_boundaries for each row

        # Compute the compression_rate only on non-padded positions for each row in the batch
        compression_rate = (num_non_padded_positions_per_row / sum_hard_boundaries_non_padded_per_row).mean()
        p_ones = (sum_hard_boundaries_non_padded_per_row / num_non_padded_positions_per_row).mean()


        return compression_rate, p_ones



    def compute_boundaries_in_parallel(self, hidden, dtype, boundary_predictor, attention_mask, device):
        stats = {}
        loss_boundaries = torch.tensor(0, dtype=dtype, device=device)
        residual = None

        # Process input with Transformer blocks
        for i in range(len(self.layers)):
            if i == 1:  # Downsampling
                #residual = hidden
                residual = hidden.clone()

                soft_boundaries, hard_boundaries = boundary_predictor(hidden)

                    # B x T
                hidden = downsample(
                    boundaries=hard_boundaries,
                    hidden=hidden,
                    null_group=self.null_group,
                )
                hidden = self.down_ln(hidden)

                # Shortening stats
                # compute stats over non-padded tokens
                compression_rate, p_ones = self.compute_compression_rate(hard_boundaries, attention_mask)
                stats['p_ones'] = p_ones.item()
                stats['compression_rate'] = compression_rate.item()
                stats['loss_boundaries'] = loss_boundaries.item()
                #Shortened length might not really reflect true length with padding
                stats['shortened_length'] = hidden.size(0)

            elif i == 2:  # Upsampling
                back_hidden = upsample(
                    boundaries=hard_boundaries,
                    shortened_hidden=hidden,
                )

                hidden = back_hidden + residual

            # Out of downsample / upsample -> regular Transformer blocks
            layers = self.layers[i]

            hidden = self._forward(
                core_input=hidden,
                layers=layers)

        return hidden, stats, soft_boundaries, hard_boundaries

    def forward(self, batch, task):
        """
        Data: Batch Size x Sequence length  --> Sequence length x Batch Size
        Attention_mask: Batch Size x Sequence length  --> Batch Size x Sequence length
        """
        data = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # In each batch, get all the unique script ids and check that they are contained in script ids
        unique_script_ids = torch.unique(data[:, 0])
        assert all(value in self.script_to_id.keys() for value in unique_script_ids.tolist())

        # Group input ids by script_ids
        chunks = {}
        for value in unique_script_ids:
            # select tensors that have the same value (script id) in the first index
            script_indices = data[:, 0] == value

            # (T X B)
            script_input_ids = data[script_indices][:, 1:]  # Input_ids for this script, remove script ids
            script_attention_mask = attention_mask[script_indices]  # Attention mask for this script, no script ids here
            chunks[self.script_to_id[value.item()]] = {"input_ids": script_input_ids.T, "attention_mask": script_attention_mask} # We don't transpose attention mask here, because the operations we need the masks for require that this dimension is retained

        # Track boundaries statistics
        overall_stats = {}

        final_logits, final_labels,  final_hidden, final_loss_boundaries = [], [], [], []
        for script, batch_dict in chunks.items():
            attn_m = batch_dict["attention_mask"]
            # We shift the input ids by 1 when computing the loss
            target_ids = batch_dict["input_ids"].clone()
            tgt_len =  target_ids.size(0)

            # Get embeddings
            embeddings = self.drop(self.word_emb(batch_dict["input_ids"]))

            # (Tokenization happens here) Downsample and upsample representations
            hidden, stats, soft_boundaries, hard_boundaries = self.compute_boundaries_in_parallel(embeddings,
                            dtype=data.dtype,
                            boundary_predictor=self.script_to_bp_layers[script],
                            device=data.device,
                            attention_mask=batch_dict["attention_mask"])
            # Calculate boundary loss here
            soft_boundaries = soft_boundaries[:, -tgt_len:]
            hard_boundaries = hard_boundaries[:, -tgt_len:]
            if task == "LM":
                loss_boundaries = self.script_to_bp_layers[script].calc_loss(
                                preds=hard_boundaries, gt=None
                            )
            else:
                # check the shape of the attention mask
                loss_boundaries = self.script_to_bp_layers[script].calc_loss_without_padding(preds=hard_boundaries, gt=None, attention_mask=batch_dict["attention_mask"])

            # Get boundaries stats
            # Not used for now
            bp_stats = self.script_to_bp_layers[script].calc_stats(
                        hard_boundaries, (batch_dict["input_ids"] == 0)[-tgt_len:].transpose(0, 1)
                    )
            for k, v in stats.items():
                    overall_stats[f'{script}_{k}'] = v

            overall_stats[f'{script}_loss_boundaries'] = loss_boundaries.item()

            # Get logits
            logit = self.final_cast(hidden)
            shift_logits = logit[:-1].contiguous()
            shift_labels = target_ids[1:].contiguous()

            final_logits.append(shift_logits)
            final_labels.append(shift_labels)
            final_loss_boundaries.append(loss_boundaries)

            if task != "LM":
                final_hidden.append(hidden)

        # concatenate and compute loss
        final_logits = torch.cat(final_logits, dim=1)
        final_labels = torch.cat(final_labels, dim=1)
        loss = self.crit(final_logits.view(-1, final_logits.size(-1)), final_labels.view(-1))

        if task == "LM":
            return loss, overall_stats, final_loss_boundaries, final_logits
        else:
            return torch.cat(final_hidden, dim=1), overall_stats, final_loss_boundaries

class MagnetAverageSingleInputWithPadding(nn.Module):
    """
    Sequence classification over Single Inputs sequences.
    We take the average of token-level representations without including padded tokens

    """
    # compute loss over non-padded tokens
    def __init__(self, num_labels, pretrained_mem_transformer):
        super(MagnetAverageSingleInputWithPadding, self).__init__()
        self.memtransformer = pretrained_mem_transformer
        self.score = nn.Linear(pretrained_mem_transformer.d_model, num_labels, bias=False)
        self.num_labels = num_labels
        self.fct = nn.CrossEntropyLoss()

    def forward(self, input_batch):
        # get the number of in
        #hidden_states, stats, boundary_loss = self.memtransformer(input_batch["input_ids"], input_batch["input_ids"].clone(), task="class")
        hidden_states, stats, boundary_loss = self.memtransformer(input_batch, task="class")
        # Compute mean without considering padding

        hidden_states = hidden_states.permute(1, 0, 2)
        hidden_states = compute_mean_with_padding(hidden_states, input_batch["attention_mask"])

        #hidden_states = torch.mean(hidden_states, dim=0)
        logits = self.score(hidden_states)
        loss = self.fct(logits.view(-1, self.num_labels), input_batch["labels"].view(-1))

        return loss, logits, stats, boundary_loss