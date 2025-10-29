import torch

from collections import namedtuple
from backend.text_processing import parsing, emphasis
from backend import memory_management
from backend.args import dynamic_args

from modules.shared import opts


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class T5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, emphasis_name="Original", min_length=256, end_with_pad=False):
        super().__init__()

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer

        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        self.min_length = min_length
        self.end_with_pad = end_with_pad
        self.id_end = 1
        self.id_pad = 0

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}

        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def encode_with_transformers(self, tokens, attention_mask=None):
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)
        self.text_encoder.shared.to(device=device, dtype=torch.float32)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)
            z = self.text_encoder(input_ids=tokens, attention_mask=attention_mask)
        else:
            z = self.text_encoder(input_ids=tokens)
        return z

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens = chunk.tokens + [self.id_end]
            chunk.multipliers = chunk.multipliers + [1.0]
            current_chunk_length = len(chunk.tokens)

            token_count += current_chunk_length
            remaining_count = self.min_length - current_chunk_length

            if remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    def __call__(self, texts, return_attention_mask=False):
        zs = []
        attention_masks = []
        cache = {}

        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        target_min_length = dynamic_args.get('t5_min_length', self.min_length)
        if isinstance(target_min_length, (int, float)):
            target_min_length = max(1, int(target_min_length))
        else:
            target_min_length = self.min_length
        if target_min_length != self.min_length:
            self.min_length = target_min_length

        for line in texts:
            if line in cache:
                line_z_values, line_attention_masks = cache[line]
            else:
                chunks, token_count = self.tokenize_line(line)
                line_z_values = []
                line_attention_masks = []

                #   pad all chunks to length of longest chunk
                max_tokens = 0
                for chunk in chunks:
                    max_tokens = max (len(chunk.tokens), max_tokens)

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers
                    actual_token_count = len(tokens)

                    remaining_count = max_tokens - len(tokens)
                    if remaining_count > 0:
                        tokens += [self.id_pad] * remaining_count
                        multipliers += [1.0] * remaining_count

                    attention_mask = torch.zeros(len(tokens), dtype=torch.bool)
                    if remaining_count > 0:
                        attention_mask[:actual_token_count + 1] = True
                    else:
                        attention_mask[:] = True

                    z = self.process_tokens([tokens], [multipliers], [attention_mask])[0]
                    line_z_values.append(z)
                    line_attention_masks.append(attention_mask)

                cache[line] = (line_z_values, line_attention_masks)

            zs.extend(line_z_values)
            attention_masks.extend(line_attention_masks)

        if return_attention_mask:
            return torch.stack(zs), torch.stack(attention_masks)
        else:
            return torch.stack(zs)

    def process_tokens(self, batch_tokens, batch_multipliers, batch_attention_masks=None):
        tokens = torch.asarray(batch_tokens)

        if batch_attention_masks is not None:
            attention_mask = torch.stack(batch_attention_masks) if len(batch_attention_masks) > 1 else batch_attention_masks[0].unsqueeze(0)
        else:
            attention_mask = None

        z = self.encode_with_transformers(tokens, attention_mask)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
