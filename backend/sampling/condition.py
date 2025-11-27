import torch
import math


def repeat_to_batch_size(tensor, batch_size):
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    elif tensor.shape[0] < batch_size:
        return tensor.repeat([math.ceil(batch_size / tensor.shape[0])] + [1] * (len(tensor.shape) - 1))[:batch_size]
    return tensor


def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


class Condition:
    def __init__(self, cond):
        self.cond = cond

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class ConditionNoiseShape(Condition):
    def process_cond(self, batch_size, device, area, **kwargs):
        data = self.cond[:, :, area[2]:area[0] + area[2], area[3]:area[1] + area[3]]
        return self._copy_with(repeat_to_batch_size(data, batch_size).to(device))


class ConditionCrossAttn(Condition):
    def __init__(self, cond, attention_mask=None):
        # cond can be a tensor or a dict with 'crossattn' and optionally 'attention_mask'
        if isinstance(cond, dict):
            self.cond = cond['crossattn']
            self.attention_mask = cond.get('attention_mask', None)
        else:
            self.cond = cond
            self.attention_mask = attention_mask

    def _copy_with(self, cond, attention_mask=None):
        return self.__class__(cond, attention_mask)

    def process_cond(self, batch_size, device, **kwargs):
        processed_cond = repeat_to_batch_size(self.cond, batch_size).to(device)
        processed_mask = None
        if self.attention_mask is not None:
            processed_mask = repeat_to_batch_size(self.attention_mask, batch_size).to(device)
        return self._copy_with(processed_cond, processed_mask)

    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4:
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        masks = [self.attention_mask]
        crossattn_max_len = self.cond.shape[1]
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])
            conds.append(c)
            masks.append(x.attention_mask)

        out = []
        out_masks = []
        has_mask = any(m is not None for m in masks)

        for i, c in enumerate(conds):
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1)
            out.append(c)

            # Handle masks if any are present
            if has_mask:
                m = masks[i]
                if m is not None:
                    if m.shape[1] < crossattn_max_len:
                        m = m.repeat(1, crossattn_max_len // m.shape[1])
                    out_masks.append(m)

        # Return dict with attention_mask if all conds had masks
        if has_mask and len(out_masks) == len(conds):
            concat_mask = torch.cat(out_masks)
            return {'crossattn': torch.cat(out), 'attention_mask': concat_mask}

        return torch.cat(out)


class ConditionConstant(Condition):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond


def compile_conditions(cond):
    if cond is None:
        return None

    if isinstance(cond, torch.Tensor):
        result = dict(
            cross_attn=cond,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond),
            )
        )
        return [result, ]

    cross_attn = cond['crossattn']
    attention_mask = cond.get('attention_mask', None)

    # Pass the full cond dict if it has attention_mask, so ConditionCrossAttn can preserve it
    if attention_mask is not None:
        result = dict(
            cross_attn=cross_attn,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cond)  # Pass full dict with attention_mask
            )
        )
    else:
        result = dict(
            cross_attn=cross_attn,
            model_conds=dict(
                c_crossattn=ConditionCrossAttn(cross_attn)
            )
        )

    if 'vector' in cond:
        result['pooled_output'] = cond['vector']
        result['model_conds']['y'] = Condition(cond['vector'])

    if 'guidance' in cond:
        result['model_conds']['guidance'] = Condition(cond['guidance'])

    return [result, ]


def compile_weighted_conditions(cond, weights):
    transposed = list(map(list, zip(*weights)))
    results = []

    for cond_pre in transposed:
        current_indices = []
        current_weight = 0
        for i, w in cond_pre:
            current_indices.append(i)
            current_weight = w

        if hasattr(cond, 'advanced_indexing'):
            feed = cond.advanced_indexing(current_indices)
        else:
            feed = cond[current_indices]

        h = compile_conditions(feed)
        h[0]['strength'] = current_weight
        results += h

    return results