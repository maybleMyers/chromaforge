import sys
import time
import psutil
import torch
import platform

from enum import Enum
from backend import stream, utils
from backend.args import args
# Block swapping imports removed - will be replaced with sd-scripts compatible implementation


cpu = torch.device('cpu')


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine device state
cpu_state = CPUState.GPU
total_vram = 0

lowvram_available = True
xpu_available = False

if args.pytorch_deterministic:
    print("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_enabled = False
if args.directml is not None:
    import torch_directml
    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    print("Using directml with device: {}".format(torch_directml.device_name(device_index)))

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

if args.always_cpu:
    cpu_state = CPUState.CPU


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        else:
            return torch.device(torch.cuda.current_device())


def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024  # TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_torch = mem_reserved
            mem_total = torch.xpu.get_device_properties(dev).total_memory
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total


total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = psutil.virtual_memory().total / (1024 * 1024)
print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))

try:
    print("pytorch version: {}".format(torch.version.__version__))
except:
    pass

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

if directml_enabled:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            print("xformers version: {}".format(XFORMERS_VERSION))
            if XFORMERS_VERSION.startswith("0.0.18"):
                print("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                print("Please downgrade or upgrade xformers to a different version.\n")
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False


def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


ENABLE_PYTORCH_ATTENTION = False
if args.attention_pytorch:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

VAE_DTYPES = [torch.float32]

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and args.attention_split == False and args.attention_quad == False:
                ENABLE_PYTORCH_ATTENTION = True
            if torch.cuda.is_bf16_supported() and torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 8:
                VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES
    if is_intel_xpu():
        if args.attention_split == False and args.attention_quad == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if is_intel_xpu():
    VAE_DTYPES = [torch.bfloat16] + VAE_DTYPES

if args.vae_in_cpu:
    VAE_DTYPES = [torch.float32]

VAE_ALWAYS_TILED = False

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

FORCE_FP32 = False
FORCE_FP16 = False
if args.all_in_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.all_in_fp16:
    print("Forcing FP16.")
    FORCE_FP16 = True

# Block swapping configuration - will be implemented with sd-scripts approach

ALWAYS_VRAM_OFFLOAD = args.always_offload_from_vram
if ALWAYS_VRAM_OFFLOAD:
    print("Always offload VRAM")

PIN_SHARED_MEMORY = args.pin_shared_memory
if PIN_SHARED_MEMORY:
    print("Always pin shared GPU memory")


# Compatibility layer for extensions expecting old VRAM system
class VRAMState(Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5

vram_state = VRAMState.NORMAL_VRAM  # Default state for compatibility

def set_vram_state_from_blocks(blocks):
    """Compatibility function for GUI integration - no longer needed with sd-scripts approach."""
    global vram_state
    # The new sd-scripts approach doesn't use vram_state, but we keep this for GUI compatibility
    if blocks == 0:
        vram_state = VRAMState.HIGH_VRAM
    elif blocks <= 10:
        vram_state = VRAMState.NORMAL_VRAM
    elif blocks <= 25:
        vram_state = VRAMState.LOW_VRAM
    else:
        vram_state = VRAMState.NO_VRAM
    # Note: This function only updates the legacy vram_state variable for compatibility.
    # The actual block swapping is handled by the sd-scripts approach in LoadedModel.model_load()

current_loaded_models = []


def state_dict_size(sd, exclude_device=None):
    module_mem = 0
    for k in sd:
        t = sd[k]
        if exclude_device is not None:
            if t.device == exclude_device:
                continue
        module_mem += t.nelement() * t.element_size()
    return module_mem


def state_dict_parameters(sd):
    module_mem = 0
    for k, v in sd.items():
        module_mem += v.nelement()
    return module_mem


def state_dict_dtype(state_dict):
    """Determine the dtype of a state dict."""
    for k, v in state_dict.items():
        if hasattr(v, 'gguf_cls'):
            return 'gguf'
        if 'bitsandbytes__nf4' in k:
            return 'nf4'
        if 'bitsandbytes__fp4' in k:
            return 'fp4'

    dtype_counts = {}

    for tensor in state_dict.values():
        dtype = tensor.dtype
        if dtype in dtype_counts:
            dtype_counts[dtype] += 1
        else:
            dtype_counts[dtype] = 1

    major_dtype = None
    max_count = 0

    for dtype, count in dtype_counts.items():
        if count > max_count:
            max_count = count
            major_dtype = dtype

    return major_dtype


def module_size(module, exclude_device=None, include_device=None, return_split=False):
    module_mem = 0
    weight_mem = 0
    weight_patterns = ['weight']

    for k, p in module.named_parameters():
        t = p.data

        if exclude_device is not None:
            if t.device == exclude_device:
                continue

        if include_device is not None:
            if t.device != include_device:
                continue

        element_size = t.element_size()

        if getattr(p, 'quant_type', None) in ['fp4', 'nf4']:
            if element_size > 1:
                element_size = 0.55
            else:
                element_size = 1.1

        module_mem += t.nelement() * element_size
        if k in weight_patterns:
            weight_mem += t.nelement() * element_size

    if return_split:
        return module_mem, weight_mem, module_mem - weight_mem
    return module_mem


# Block swapping analysis functions removed - will be replaced with sd-scripts approach


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        # Old block swapping attributes removed

    # Old get_blocks_to_swap method removed

    def model_load(self, model_gpu_memory_when_using_cpu_swap=-1):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.forge_patch_model(self.device)
            self.model.current_device = self.model.load_device
        except Exception as e:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        # SD-scripts compatible block swapping implementation
        try:
            import sys
            if 'modules.shared' in sys.modules:
                from modules import shared
                if hasattr(shared, 'opts') and shared.opts is not None:
                    blocks_to_swap = getattr(shared.opts, 'blocks_to_swap', 0)
                else:
                    blocks_to_swap = 0
            else:
                # Fallback to command line args
                blocks_to_swap = getattr(args, 'blocks_to_swap', 0)
        except (ImportError, AttributeError):
            blocks_to_swap = getattr(args, 'blocks_to_swap', 0)

        # Enable sd-scripts compatible block swapping if model supports it
        if blocks_to_swap > 0 and hasattr(self.real_model, 'enable_block_swap'):
            try:
                print(f"[SD-Scripts Block Swapping] Found enable_block_swap method in {self.real_model.__class__.__name__}")
                self.real_model.enable_block_swap(blocks_to_swap, self.device)
                print(f"[SD-Scripts Block Swapping] Successfully enabled block swapping with {blocks_to_swap} blocks")

                # Call prepare_block_swap_before_forward if available
                if hasattr(self.real_model, 'prepare_block_swap_before_forward'):
                    self.real_model.prepare_block_swap_before_forward()
                    print(f"[SD-Scripts Block Swapping] Block devices prepared for forward pass")
            except Exception as e:
                print(f"[SD-Scripts Block Swapping] Failed to enable block swapping: {e}")
        elif blocks_to_swap > 0:
            print(f'[SD-Scripts Block Swapping] Model {self.real_model.__class__.__name__} does not support enable_block_swap method')
        else:
            print('[SD-Scripts Block Swapping] Disabled - all blocks loaded to GPU')

        return self.real_model

    # Old forward hooks methods removed

    def model_unload(self, avoid_model_moving=False):
        if avoid_model_moving:
            self.model.forge_unpatch_model()
        else:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model


current_inference_memory = 1024 * 1024 * 1024


def minimum_inference_memory():
    global current_inference_memory
    return current_inference_memory


def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = mem_free_xpu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def free_memory(memory_required, device, keep_loaded=[], free_all=False):
    # Unload abandoned models
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            current_loaded_models.pop(i).model_unload(avoid_model_moving=True)

    if free_all:
        memory_required = 1e30
        print(f"[Unload] Trying to free all memory for {device} with {len(keep_loaded)} models keep loaded ... ", end="")
    else:
        print(f"[Unload] Trying to free {memory_required / (1024 * 1024):.2f} MB for {device} with {len(keep_loaded)} models keep loaded ... ", end="")

    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        free_memory_current = get_free_memory(device)
        print(f"Current free memory is {free_memory_current / (1024 * 1024):.2f} MB ... ", end="")
        if free_memory_current > memory_required:
            break

        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                m = current_loaded_models.pop(i)
                print(f"Unload model {m.model.model.__class__.__name__} ", end="")
                m.model_unload()
                del m
                unloaded_model = True

    if unloaded_model:
        soft_empty_cache()

    print('Done.')


def load_models_gpu(models, memory_required=0, hard_memory_preservation=0):
    execution_start_time = time.perf_counter()
    memory_to_free = max(minimum_inference_memory(), memory_required) + hard_memory_preservation

    models_to_load = []
    models_already_loaded = []

    for x in models:
        loaded_model = LoadedModel(x)
        if loaded_model in current_loaded_models:
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
        else:
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(memory_to_free, d, models_already_loaded)

        moving_time = time.perf_counter() - execution_start_time
        if moving_time > 0.1:
            print(f'Memory cleanup has taken {moving_time:.2f} seconds')
        return

    for loaded_model in models_to_load:
        torch_dev = loaded_model.model.load_device
        current_free_mem = get_free_memory(torch_dev)

        # Block swapping info will be handled by sd-scripts approach
        print(f"[Memory Management] Target: {loaded_model.model.model.__class__.__name__}, "
              f"Free GPU: {current_free_mem / (1024 * 1024):.2f} MB")

        loaded_model.model_load()
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    print(f'Moving model(s) has taken {moving_time:.2f} seconds')


def load_model_gpu(model):
    return load_models_gpu([model])


def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def unload_all_models():
    free_memory(1e30, get_torch_device(), free_all=True)


# Keep remaining utility functions from original
def is_device_cpu(device):
    return hasattr(device, 'type') and device.type == 'cpu'


def is_device_mps(device):
    return hasattr(device, 'type') and device.type == 'mps'


def is_device_cuda(device):
    return hasattr(device, 'type') and device.type == 'cuda'


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def unet_offload_device():
    return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    cpu_dev = torch.device("cpu")

    if ALWAYS_VRAM_OFFLOAD:
        return cpu_dev

    model_size = dtype.itemsize * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)

    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def text_encoder_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def text_encoder_offload_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_device():
    if args.vae_in_cpu:
        return torch.device("cpu")
    return get_torch_device()


def vae_offload_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def intermediate_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def dtype_size(dtype):
    """Get the size in bytes of a dtype."""
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    """Get the best dtype for UNet models."""
    if args.unet_in_bf16:
        return torch.bfloat16
    if args.unet_in_fp16:
        return torch.float16
    if args.unet_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    if args.unet_in_fp8_e5m2:
        return torch.float8_e5m2

    for candidate in supported_dtypes:
        if candidate == torch.float16:
            if should_use_fp16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
                return candidate
        if candidate == torch.bfloat16:
            if should_use_bf16(device, model_params=model_params, prioritize_performance=True, manual_cast=True):
                return candidate

    return torch.float32


def get_computation_dtype(inference_device, parameters=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
    """Get the best computation dtype."""
    for candidate in supported_dtypes:
        if candidate == torch.float16:
            if should_use_fp16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
                return candidate
        if candidate == torch.bfloat16:
            if should_use_bf16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
                return candidate
    return torch.float32


def text_encoder_dtype(device=None):
    """Get the best dtype for text encoders."""
    if args.clip_in_fp8_e4m3fn:
        return torch.float8_e4m3fn
    elif args.clip_in_fp8_e5m2:
        return torch.float8_e5m2
    elif args.clip_in_fp16:
        return torch.float16
    elif args.clip_in_fp32:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16
    return torch.float16


def vae_dtype(device=None, allowed_dtypes=[]):
    """Get the best dtype for VAE models."""
    global VAE_DTYPES
    if args.vae_in_fp16:
        return torch.float16
    elif args.vae_in_bf16:
        return torch.bfloat16
    elif args.vae_in_fp32:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device, prioritize_performance=False):
            return d
        if d in VAE_DTYPES:
            return d

    return VAE_DTYPES[0]


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Determine if FP16 should be used."""
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if torch.version.hip:
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major >= 8:
        return True

    if props.major < 6:
        return False

    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            if manual_cast:
                # For storage dtype
                free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
                if (not prioritize_performance) or model_params * 4 > free_model_memory:
                    return True
            else:
                # For computation dtype
                return False  # Flux on 1080 can store model in fp16 to reduce swap, but computation must be fp32, otherwise super slow.

    if props.major < 7:
        return False

    # FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    """Determine if BF16 should be used."""
    if device is not None:
        if is_device_cpu(device):  # TODO ? bf16 works on CPU but is extremely slow
            return False

    if device is not None:
        if is_device_mps(device):
            return True

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return True

    if device is None:
        device = torch.device("cuda")

    props = torch.cuda.get_device_properties(device)
    if props.major >= 8:
        return True

    if torch.cuda.is_bf16_supported():
        # This device is an old enough device but bf16 somewhat reports supported.
        # So in this case bf16 should only be used as storge dtype
        if manual_cast:
            # For storage dtype
            free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
            if (not prioritize_performance) or model_params * 4 > free_model_memory:
                return True

    return False


def get_torch_device_name(device):
    """Get a descriptive name for a torch device."""
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


def xformers_enabled():
    """Check if xformers is enabled."""
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    """Check if xformers is enabled for VAE."""
    enabled = xformers_enabled()
    if not enabled:
        return False
    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    """Check if PyTorch attention is enabled."""
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    """Check if PyTorch flash attention is available."""
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        # TODO: more reliable way of checking for flash attention?
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
        if is_intel_xpu():
            return True
    return False


def sage_attention_enabled():
    """Check if sage attention is enabled."""
    return args.use_sage_attention


def flash_attention_enabled():
    """Check if flash attention is enabled."""
    return args.use_flash_attention


def force_upcast_attention_dtype():
    """Get forced upcast attention dtype if any."""
    upcast = args.force_upcast_attention
    try:
        if platform.mac_ver()[0] in ['14.5']:  # black image bug on OSX Sonoma 14.5
            upcast = True
    except:
        pass
    if upcast:
        return torch.float32
    else:
        return None


def force_channels_last():
    """Check if channels last format should be forced."""
    if args.force_channels_last:
        return True
    return False


def can_install_bnb():
    """Check if bitsandbytes can be installed."""
    try:
        if not torch.cuda.is_available():
            return False
        cuda_version = tuple(int(x) for x in torch.version.cuda.split('.'))
        if cuda_version >= (11, 7):
            return True
        return False
    except:
        return False


def device_supports_non_blocking(device):
    """Check if device supports non-blocking transfers."""
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    if is_intel_xpu():
        return False
    if args.pytorch_deterministic:  # TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_enabled:
        return False
    return True


def device_should_use_non_blocking(device):
    """Check if device should use non-blocking transfers."""
    if not device_supports_non_blocking(device):
        return False
    return False
    # return True #TODO: figure out why this causes memory issues on Nvidia and possibly others


def cast_to_device(tensor, device, dtype, copy=False):
    """Cast tensor to device and dtype with proper handling."""
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, 'type') and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    non_blocking = device_should_use_non_blocking(device)

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy, non_blocking=non_blocking)
            return tensor.to(device, copy=copy, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
        else:
            return tensor.to(device, non_blocking=non_blocking).to(dtype, non_blocking=non_blocking)
    else:
        return tensor.to(device, dtype, copy=copy, non_blocking=non_blocking)


def supports_dtype(device, dtype):
    """Check if device supports dtype."""
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def supports_cast(device, dtype):
    """Check if device supports casting to dtype."""
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_enabled:  # TODO: test this
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    """Pick appropriate weight dtype."""
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype


def get_autocast_device(dev):
    """Get autocast device type string."""
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"


print(f"VAE dtype preferences: {VAE_DTYPES} -> {vae_dtype()}")

# SD-scripts compatible block swapping status
try:
    import sys
    if 'modules.shared' in sys.modules:
        from modules import shared
        if hasattr(shared, 'opts') and shared.opts is not None:
            current_blocks_to_swap = getattr(shared.opts, 'blocks_to_swap', 0)
        else:
            current_blocks_to_swap = 0
    else:
        current_blocks_to_swap = getattr(args, 'blocks_to_swap', 0)
except (ImportError, AttributeError):
    current_blocks_to_swap = getattr(args, 'blocks_to_swap', 0)

print(f"SD-Scripts Block Swapping: {'Enabled' if current_blocks_to_swap > 0 else 'Disabled'} ({current_blocks_to_swap} blocks)")

try:
    torch_device_name = get_torch_device_name(get_torch_device())
    print("Device: {}".format(torch_device_name))
except:
    torch_device_name = ''
    print("Could not pick default device.")

# Signal for cache clearing
signal_empty_cache = False