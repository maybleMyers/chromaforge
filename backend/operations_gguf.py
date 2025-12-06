import gguf
import torch


quants_mapping = {
    gguf.GGMLQuantizationType.Q2_K: gguf.Q2_K,
    gguf.GGMLQuantizationType.Q3_K: gguf.Q3_K,
    gguf.GGMLQuantizationType.Q4_0: gguf.Q4_0,
    gguf.GGMLQuantizationType.Q4_K: gguf.Q4_K,
    gguf.GGMLQuantizationType.Q4_1: gguf.Q4_1,
    gguf.GGMLQuantizationType.Q5_0: gguf.Q5_0,
    gguf.GGMLQuantizationType.Q5_1: gguf.Q5_1,
    gguf.GGMLQuantizationType.Q5_K: gguf.Q5_K,
    gguf.GGMLQuantizationType.Q6_K: gguf.Q6_K,
    gguf.GGMLQuantizationType.Q8_0: gguf.Q8_0,
    gguf.GGMLQuantizationType.BF16: gguf.BF16,
}


class ParameterGGUF(torch.nn.Parameter):
    def __init__(self, tensor=None, requires_grad=False, no_init=False):
        super().__init__()
        if no_init:
            return

        self.gguf_cls = quants_mapping.get(tensor.tensor_type, None)
        self.real_shape = torch.Size(reversed(list(tensor.shape)))
        self.computation_dtype = torch.float16
        self.baked = False
        return

    @property
    def shape(self):
        return self.real_shape

    def __new__(cls, tensor=None, requires_grad=False, no_init=False):
        return super().__new__(cls, torch.tensor(tensor.data), requires_grad=requires_grad)

    def dequantize_as_pytorch_parameter(self):
        if self.gguf_cls is not None and hasattr(self.gguf_cls, 'bake'):
            self.gguf_cls.bake(self)
        dequantized = dequantize_tensor(self)
        # Ensure it's a plain tensor before creating Parameter
        if not isinstance(dequantized, torch.Tensor) or isinstance(dequantized, torch.nn.Parameter):
            dequantized = dequantized.data.clone()
        return torch.nn.Parameter(dequantized.clone().detach(), requires_grad=False)

    def copy_with_data(self, data):
        new = ParameterGGUF(data, no_init=True)
        new.gguf_cls = self.gguf_cls
        new.real_shape = self.real_shape
        new.computation_dtype = self.computation_dtype
        new.baked = self.baked
        return new

    def to(self, *args, **kwargs):
        return self.copy_with_data(self.data.to(*args, **kwargs))

    def pin_memory(self, device=None):
        return self.copy_with_data(torch.Tensor.pin_memory(self, device=device))


def dequantize_tensor(tensor):
    if tensor is None:
        return None

    if not hasattr(tensor, 'gguf_cls'):
        return tensor

    gguf_cls = tensor.gguf_cls

    if gguf_cls is None:
        return tensor

    # Handle both old and new gguf library API
    if hasattr(gguf_cls, 'dequantize_pytorch'):
        result = gguf_cls.dequantize_pytorch(tensor)
    else:
        # New gguf library uses numpy-based dequantize
        import numpy as np
        # Get the raw data as numpy array
        data_np = tensor.data.cpu().numpy()
        # Dequantize using the gguf library
        dequantized_np = gguf_cls.dequantize(data_np)
        # Get the real shape and reshape
        real_shape = tensor.real_shape if hasattr(tensor, 'real_shape') else tensor.shape
        dequantized_np = dequantized_np.reshape(real_shape)
        # Convert back to torch tensor
        result = torch.from_numpy(dequantized_np.copy()).to(tensor.computation_dtype if hasattr(tensor, 'computation_dtype') else torch.float16)

    # Ensure we return a plain tensor, not a ParameterGGUF subclass
    if hasattr(result, 'gguf_cls'):
        result = result.data.clone()
    return result