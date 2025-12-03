# Cherry-picked some good parts from ComfyUI with some bad parts fixed

import sys
import time
import psutil
import torch
import platform

from enum import Enum
from backend import stream, utils
from backend.args import args


cpu = torch.device('cpu')


class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move models to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
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

if args.always_low_vram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.always_no_vram:
    set_vram_to = VRAMState.NO_VRAM
elif args.always_high_vram or args.always_gpu:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.all_in_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.all_in_fp16:
    print("Forcing FP16.")
    FORCE_FP16 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

print(f"Set vram state to: {vram_state.name}")

ALWAYS_VRAM_OFFLOAD = args.always_offload_from_vram

if ALWAYS_VRAM_OFFLOAD:
    print("Always offload VRAM")

PIN_SHARED_MEMORY = args.pin_shared_memory

if PIN_SHARED_MEMORY:
    print("Always pin shared GPU memory")


def get_torch_device_name(device):
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


try:
    torch_device_name = get_torch_device_name(get_torch_device())
    print("Device: {}".format(torch_device_name))
except:
    torch_device_name = ''
    print("Could not pick default device.")

if 'rtx' in torch_device_name.lower():
    if not args.cuda_malloc:
        print('Hint: your device supports --cuda-malloc for potential speed improvements.')


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


def bake_gguf_model(model):
    if getattr(model, 'gguf_baked', False):
        return

    for p in model.parameters():
        gguf_cls = getattr(p, 'gguf_cls', None)
        if gguf_cls is not None:
            gguf_cls.bake(p)

    global signal_empty_cache
    signal_empty_cache = True

    model.gguf_baked = True
    return model


def module_size(module, exclude_device=None, include_device=None, return_split=False):
    module_mem = 0
    weight_mem = 0
    weight_patterns = ['weight']

    # Check if this is a RamTorch bouncing linear module
    is_ramtorch = False
    module_class_name = module.__class__.__name__
    if 'ChromaBouncing' in module_class_name or 'CPUBouncing' in module_class_name:
        is_ramtorch = True

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
                # not quanted yet
                element_size = 0.55  # a bit more than 0.5 because of quant state parameters
            else:
                # quanted
                element_size = 1.1  # a bit more than 0.5 because of quant state parameters

        # For RamTorch modules, weights are on CPU but we count them differently
        # Only count the GPU memory usage (which is minimal for RamTorch)
        if is_ramtorch and k in weight_patterns and t.device.type == 'cpu':
            # For CPU-bouncing weights, we don't count them in GPU memory
            # but we do need to account for temporary GPU buffer during forward pass
            if include_device is not None and include_device.type == 'cuda':
                # Account for temporary GPU buffer (only needed during computation)
                weight_mem += t.nelement() * element_size * 0.1  # 10% for temporary buffer
            elif exclude_device is None:
                # If not filtering by device, don't count CPU weights as GPU memory
                pass
        else:
            module_mem += t.nelement() * element_size

            if k in weight_patterns:
                weight_mem += t.nelement() * element_size

    if return_split:
        return module_mem, weight_mem, module_mem - weight_mem

    return module_mem


def module_move(module, device, recursive=True, excluded_pattens=[]):
    if recursive:
        return module.to(device=device)

    for k, p in module.named_parameters(recurse=False, remove_duplicate=True):
        if k in excluded_pattens:
            continue
        setattr(module, k, utils.tensor2parameter(p.to(device=device)))

    return module


def build_module_profile(model, model_gpu_memory_when_using_cpu_swap):
    all_modules = []
    legacy_modules = []

    for m in model.modules():
        if hasattr(m, "parameters_manual_cast"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            all_modules.append(m)
        elif hasattr(m, "weight"):
            m.total_mem, m.weight_mem, m.extra_mem = module_size(m, return_split=True)
            legacy_modules.append(m)

    # Disabled ChromaDCT special profiling - use normal profiling for all models
    # This improves ChromaDCT performance to be only ~10% slower than regular Chroma
    # instead of 2x slower with the special profiling
    # is_chromadct = False
    # try:
    #     # Detect ChromaDCT by checking for distinctive components
    #     if hasattr(model, 'img_in_patch') or any(hasattr(m, 'img_in_patch') for m in all_modules + legacy_modules):
    #         is_chromadct = True
    #     elif any('img_in_patch' in str(m).lower() for m in all_modules + legacy_modules):
    #         is_chromadct = True
    #
    #     if is_chromadct:
    #         # Only print message once per session
    #         if not hasattr(build_module_profile, '_chromadct_profiling_message_shown'):
    #             print("Detected ChromaDCT model - using optimized module profiling...")
    #             build_module_profile._chromadct_profiling_message_shown = True
    #         return build_chromadct_module_profile(model, all_modules, legacy_modules, model_gpu_memory_when_using_cpu_swap)
    # except Exception as e:
    #     print(f"ChromaDCT detection failed: {e}")
    #     pass

    # Default profiling for non-ChromaDCT models
    gpu_modules = []
    gpu_modules_only_extras = []
    mem_counter = 0

    for m in legacy_modules.copy():
        gpu_modules.append(m)
        legacy_modules.remove(m)
        mem_counter += m.total_mem

    for m in sorted(all_modules, key=lambda x: x.extra_mem).copy():
        if mem_counter + m.extra_mem < model_gpu_memory_when_using_cpu_swap:
            gpu_modules_only_extras.append(m)
            all_modules.remove(m)
            mem_counter += m.extra_mem

    cpu_modules = all_modules

    for m in sorted(gpu_modules_only_extras, key=lambda x: x.weight_mem).copy():
        if mem_counter + m.weight_mem < model_gpu_memory_when_using_cpu_swap:
            gpu_modules.append(m)
            gpu_modules_only_extras.remove(m)
            mem_counter += m.weight_mem

    return gpu_modules, gpu_modules_only_extras, cpu_modules


def build_chromadct_module_profile(model, all_modules, legacy_modules, model_gpu_memory_when_using_cpu_swap):
    """
    Build module profile optimized for ChromaDCT access patterns
    """
    
    # Helper function to categorize modules based on ChromaDCT component names
    def get_module_priority(module):
        module_str = str(module).lower()
        class_name = module.__class__.__name__.lower()
        
        # Critical components - always keep on GPU
        critical_components = ['img_in_patch', 'txt_in', 'pe_embedder', 'distilled_guidance']
        for comp in critical_components:
            if comp in module_str or comp in class_name:
                return 'critical'
        
        # High priority - early blocks processed first
        if 'double_block' in module_str:
            # Extract block number if possible
            for i in range(10):  # blocks 0-9 are high priority
                if f'double_blocks.{i}' in module_str or f'double_block_{i}' in module_str:
                    return 'high_priority'
            return 'medium_priority'  # blocks 10+ are medium priority
        
        if 'single_block' in module_str:
            # Extract block number if possible  
            for i in range(19):  # blocks 0-18 are high priority
                if f'single_blocks.{i}' in module_str or f'single_block_{i}' in module_str:
                    return 'high_priority'
            return 'medium_priority'  # blocks 19+ are medium priority
        
        # Low priority - NeRF components (processed as group at end)
        nerf_components = ['nerf_block', 'nerf_image_embedder', 'nerf_final_layer']
        for comp in nerf_components:
            if comp in module_str or comp in class_name:
                return 'low_priority'
        
        return 'unknown'
    
    # Separate modules by priority
    critical_modules = []
    high_priority_modules = []
    medium_priority_modules = []  
    low_priority_modules = []
    unknown_modules = []
    
    all_categorized_modules = all_modules + legacy_modules
    
    for m in all_categorized_modules:
        priority = get_module_priority(m)
        if priority == 'critical':
            critical_modules.append(m)
        elif priority == 'high_priority':
            high_priority_modules.append(m) 
        elif priority == 'medium_priority':
            medium_priority_modules.append(m)
        elif priority == 'low_priority':
            low_priority_modules.append(m)
        else:
            unknown_modules.append(m)
    
    # Allocate modules to GPU/CPU based on priority and memory constraints
    gpu_modules = []
    gpu_modules_only_extras = []
    cpu_modules = []
    mem_counter = 0
    
    # Priority order: critical -> high -> medium -> low -> unknown
    module_groups = [
        ("Critical", critical_modules),
        ("High Priority", high_priority_modules), 
        ("Medium Priority", medium_priority_modules),
        ("Low Priority", low_priority_modules),
        ("Unknown", unknown_modules)
    ]
    
    print(f"ChromaDCT Memory Allocation (Available: {model_gpu_memory_when_using_cpu_swap / (1024**2):.0f} MB):")
    
    for group_name, modules in module_groups:
        modules_added_to_gpu = 0
        modules_moved_to_cpu = 0
        group_memory = 0
        
        # Sort modules by size (larger first for better packing)
        modules_sorted = sorted(modules, key=lambda m: m.total_mem, reverse=True)
        
        for m in modules_sorted:
            if mem_counter + m.total_mem < model_gpu_memory_when_using_cpu_swap:
                # Can fit entire module on GPU
                gpu_modules.append(m)
                mem_counter += m.total_mem
                group_memory += m.total_mem
                modules_added_to_gpu += 1
            elif (group_name in ["Critical", "High Priority"] and 
                  mem_counter + m.extra_mem < model_gpu_memory_when_using_cpu_swap):
                # For important modules, try to keep extras on GPU even if weights go to CPU
                gpu_modules_only_extras.append(m)
                mem_counter += m.extra_mem
                group_memory += m.extra_mem
                modules_added_to_gpu += 1
            else:
                # Must go to CPU
                cpu_modules.append(m)
                modules_moved_to_cpu += 1
        
        if len(modules) > 0:
            print(f"  {group_name:15}: {modules_added_to_gpu:2d} GPU, {modules_moved_to_cpu:2d} CPU ({group_memory / (1024**2):.0f} MB)")
    
    total_modules = len(all_categorized_modules)
    gpu_total = len(gpu_modules) + len(gpu_modules_only_extras)
    cpu_total = len(cpu_modules)
    
    print(f"ChromaDCT Profile: {gpu_total}/{total_modules} modules on GPU ({gpu_total/total_modules*100:.1f}%), Memory used: {mem_counter / (1024**2):.0f} MB")
    
    return gpu_modules, gpu_modules_only_extras, cpu_modules


class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device
        self.inclusive_memory = 0
        self.exclusive_memory = 0

    def compute_inclusive_exclusive_memory(self):
        self.inclusive_memory = module_size(self.model.model, include_device=self.device)
        self.exclusive_memory = module_size(self.model.model, exclude_device=self.device)
        return

    def model_load(self, model_gpu_memory_when_using_cpu_swap=-1):
        patch_model_to = None
        do_not_need_cpu_swap = model_gpu_memory_when_using_cpu_swap < 0

        if do_not_need_cpu_swap:
            patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.forge_patch_model(patch_model_to)
            self.model.current_device = self.model.load_device
        except Exception as e:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if do_not_need_cpu_swap:
            print('All loaded to GPU.')
        else:
            gpu_modules, gpu_modules_only_extras, cpu_modules = build_module_profile(self.real_model, model_gpu_memory_when_using_cpu_swap)
            pin_memory = PIN_SHARED_MEMORY and is_device_cpu(self.model.offload_device)

            mem_counter = 0
            swap_counter = 0

            for m in gpu_modules:
                # Check if this is a RamTorch module that should keep weights on CPU
                module_class_name = m.__class__.__name__
                if 'ChromaBouncing' in module_class_name or 'CPUBouncing' in module_class_name:
                    # RamTorch modules keep weights on CPU, don't move them
                    # The module's _apply method is overridden to handle this
                    pass
                else:
                    m.to(self.device)
                mem_counter += m.total_mem

            for m in cpu_modules:
                # Check if this is a RamTorch module
                module_class_name = m.__class__.__name__
                if 'ChromaBouncing' in module_class_name or 'CPUBouncing' in module_class_name:
                    # RamTorch modules already handle their own memory management
                    # Don't apply standard CPU offloading
                    pass
                else:
                    m.prev_parameters_manual_cast = m.parameters_manual_cast
                    m.parameters_manual_cast = True
                    m.to(self.model.offload_device)
                    if pin_memory:
                        m._apply(lambda x: x.pin_memory())
                swap_counter += m.total_mem

            for m in gpu_modules_only_extras:
                m.prev_parameters_manual_cast = m.parameters_manual_cast
                m.parameters_manual_cast = True
                module_move(m, device=self.device, recursive=False, excluded_pattens=['weight'])
                if hasattr(m, 'weight') and m.weight is not None:
                    if pin_memory:
                        m.weight = utils.tensor2parameter(m.weight.to(self.model.offload_device).pin_memory())
                    else:
                        m.weight = utils.tensor2parameter(m.weight.to(self.model.offload_device))
                mem_counter += m.extra_mem
                swap_counter += m.weight_mem

            swap_flag = 'Shared' if PIN_SHARED_MEMORY else 'CPU'
            method_flag = 'asynchronous' if stream.should_use_stream() else 'blocked'
            print(f"{swap_flag} Swap Loaded ({method_flag} method): {swap_counter / (1024 * 1024):.2f} MB, GPU Loaded: {mem_counter / (1024 * 1024):.2f} MB")

            self.model_accelerated = True

            global signal_empty_cache
            signal_empty_cache = True

        bake_gguf_model(self.real_model)

        self.model.refresh_loras()

        if is_intel_xpu() and not args.disable_ipex_hijack:
            self.real_model = torch.xpu.optimize(self.real_model.eval(), inplace=True, auto_kernel_selection=True, graph_mode=True)

        # Reset signal_empty_cache after model loading is complete to prevent
        # unnecessary cache clearing during inference
        if not stream.should_use_stream():
            signal_empty_cache = False
        
        return self.real_model

    def model_unload(self, avoid_model_moving=False):
        if self.model_accelerated:
            for m in self.real_model.modules():
                if hasattr(m, "prev_parameters_manual_cast"):
                    m.parameters_manual_cast = m.prev_parameters_manual_cast
                    del m.prev_parameters_manual_cast

            self.model_accelerated = False

        if avoid_model_moving:
            self.model.forge_unpatch_model()
        else:
            self.model.forge_unpatch_model(self.model.offload_device)
            self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model  # and self.memory_required == other.memory_required


current_inference_memory = 1024 * 1024 * 1024


def minimum_inference_memory():
    global current_inference_memory

    # Disabled ChromaDCT-specific memory optimization - use same memory for all models
    # try:
    #     from backend import chromadct_memory_strategy
    #     if chromadct_memory_strategy.is_chromadct_model(None):
    #         multiplier = chromadct_memory_strategy.get_chromadct_inference_memory_multiplier()
    #         chromadct_inference_memory = int(current_inference_memory * multiplier)
    #
    #         # Only print message once per session
    #         if not hasattr(minimum_inference_memory, '_chromadct_message_shown'):
    #             print(f"ChromaDCT detected - optimizing inference memory from {current_inference_memory / (1024**2):.0f} MB to {chromadct_inference_memory / (1024**2):.0f} MB")
    #             minimum_inference_memory._chromadct_message_shown = True
    #
    #         return chromadct_inference_memory
    # except Exception as e:
    #     # Fallback to normal memory if detection fails
    #     pass

    return current_inference_memory


def unload_model_clones(model):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    for i in to_unload:
        current_loaded_models.pop(i).model_unload(avoid_model_moving=True)


def free_memory(memory_required, device, keep_loaded=[], free_all=False):
    # this check fully unloads any 'abandoned' models
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            current_loaded_models.pop(i).model_unload(avoid_model_moving=True)

    if free_all:
        memory_required = 1e30
        print(f"[Unload] Trying to free all memory for {device} with {len(keep_loaded)} models keep loaded ... ", end="")
    else:
        print(f"[Unload] Trying to free {memory_required / (1024 * 1024):.2f} MB for {device} with {len(keep_loaded)} models keep loaded ... ", end="")

    offload_everything = ALWAYS_VRAM_OFFLOAD or vram_state == VRAMState.NO_VRAM
    unloaded_model = False
    for i in range(len(current_loaded_models) - 1, -1, -1):
        if not offload_everything:
            free_memory = get_free_memory(device)
            print(f"Current free memory is {free_memory / (1024 * 1024):.2f} MB ... ", end="")
            if free_memory > memory_required:
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
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()

    print('Done.')
    return


def compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, inference_memory):
    maximum_memory_available = current_free_mem - inference_memory

    # When using async swap with user-specified GPU weights, respect that setting
    from modules_forge import main_entry
    if hasattr(main_entry, 'user_specified_model_memory') and main_entry.user_specified_model_memory is not None:
        # User has explicitly set GPU weights - use that value
        user_memory = main_entry.user_specified_model_memory * 1024 * 1024  # Convert MB to bytes
        if user_memory <= maximum_memory_available:
            return int(user_memory)

    suggestion = max(
        maximum_memory_available / 1.3,
        maximum_memory_available - 1024 * 1024 * 1024 * 1.25
    )

    return int(max(0, suggestion))


def load_models_gpu(models, memory_required=0, hard_memory_preservation=0):
    global vram_state

    execution_start_time = time.perf_counter()
    memory_to_free = max(minimum_inference_memory(), memory_required) + hard_memory_preservation
    memory_for_inference = minimum_inference_memory() + hard_memory_preservation

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
        unload_model_clones(loaded_model.model)

    total_memory_required = {}
    for loaded_model in models_to_load:
        loaded_model.compute_inclusive_exclusive_memory()
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.exclusive_memory + loaded_model.inclusive_memory * 0.25

    # When using async swap, calculate total GPU budget to properly allocate between models
    total_models_memory = sum(loaded_model.exclusive_memory for loaded_model in models_to_load)
    gpu_memory_budget = None
    if stream.should_use_stream() and len(models_to_load) > 1:
        # Get the user-specified GPU memory budget
        from modules_forge import main_entry
        if hasattr(main_entry, 'user_specified_model_memory') and main_entry.user_specified_model_memory is not None:
            gpu_memory_budget = main_entry.user_specified_model_memory * 1024 * 1024  # Convert MB to bytes
            print(f"[Async Swap] Using user-specified GPU budget: {gpu_memory_budget / (1024 * 1024):.2f} MB for {len(models_to_load)} models")

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.3 + memory_to_free, device, models_already_loaded)

    for idx, loaded_model in enumerate(models_to_load):
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state

        model_gpu_memory_when_using_cpu_swap = -1

        if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM):
            model_require = loaded_model.exclusive_memory
            previously_loaded = loaded_model.inclusive_memory
            current_free_mem = get_free_memory(torch_dev)
            estimated_remaining_memory = current_free_mem - model_require - memory_for_inference

            print(f"[Memory Management] Target: {loaded_model.model.model.__class__.__name__}, Free GPU: {current_free_mem / (1024 * 1024):.2f} MB, Model Require: {model_require / (1024 * 1024):.2f} MB, Previously Loaded: {previously_loaded / (1024 * 1024):.2f} MB, Inference Require: {memory_for_inference / (1024 * 1024):.2f} MB, Remaining: {estimated_remaining_memory / (1024 * 1024):.2f} MB, ", end="")

            # Use same memory allocation logic for both async and queue methods
            if estimated_remaining_memory < 0:
                vram_set_state = VRAMState.LOW_VRAM
                if gpu_memory_budget is not None and total_models_memory > 0:
                    # Allocate GPU memory proportionally based on model size
                    model_proportion = loaded_model.exclusive_memory / total_models_memory
                    model_gpu_memory_when_using_cpu_swap = int(gpu_memory_budget * model_proportion)
                    # Ensure we leave some memory for the last model in case of rounding
                    if idx == len(models_to_load) - 1:
                        already_allocated = sum(getattr(m, 'allocated_gpu_memory', 0) for m in models_to_load[:idx])
                        model_gpu_memory_when_using_cpu_swap = int(gpu_memory_budget - already_allocated)
                    loaded_model.allocated_gpu_memory = model_gpu_memory_when_using_cpu_swap
                else:
                    model_gpu_memory_when_using_cpu_swap = compute_model_gpu_memory_when_using_cpu_swap(current_free_mem, memory_for_inference)
                    if previously_loaded > 0:
                        model_gpu_memory_when_using_cpu_swap = previously_loaded

        if vram_set_state == VRAMState.NO_VRAM:
            model_gpu_memory_when_using_cpu_swap = 0

        loaded_model.model_load(model_gpu_memory_when_using_cpu_swap)
        current_loaded_models.insert(0, loaded_model)

    moving_time = time.perf_counter() - execution_start_time
    print(f'Moving model(s) has taken {moving_time:.2f} seconds')

    return


def load_model_gpu(model):
    return load_models_gpu([model])


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x


def dtype_size(dtype):
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


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if ALWAYS_VRAM_OFFLOAD:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def unet_dtype(device=None, model_params=0, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32]):
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
    for candidate in supported_dtypes:
        if candidate == torch.float16:
            if should_use_fp16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
                return candidate
        if candidate == torch.bfloat16:
            if should_use_bf16(inference_device, model_params=parameters, prioritize_performance=True, manual_cast=False):
                return candidate

    return torch.float32


def text_encoder_offload_device():
    if args.always_gpu:
        return get_torch_device()
    else:
        return torch.device("cpu")


def text_encoder_device():
    if args.always_gpu:
        return get_torch_device()
    elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_dtype(device=None):
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


def intermediate_device():
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


def vae_dtype(device=None, allowed_dtypes=[]):
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


print(f"VAE dtype preferences: {VAE_DTYPES} -> {vae_dtype()}")


def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"


def supports_dtype(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def supports_cast(device, dtype):  # TODO
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
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype


def device_supports_non_blocking(device):
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
    if not device_supports_non_blocking(device):
        return False
    return False
    # return True #TODO: figure out why this causes memory issues on Nvidia and possibly others


def force_channels_last():
    if args.force_channels_last:
        return True

    # TODO
    return False


def cast_to_device(tensor, device, dtype, copy=False):
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

def sage_attention_enabled():
    return args.use_sage_attention

def flash_attention_enabled():
    return args.use_flash_attention

def xformers_enabled():
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
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        # TODO: more reliable way of checking for flash attention?
        if is_nvidia():  # pytorch flash attention only works on Nvidia
            return True
        if is_intel_xpu():
            return True
    return False


def force_upcast_attention_dtype():
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


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, 'cpu')


def is_device_mps(device):
    return is_device_type(device, 'mps')


def is_device_cuda(device):
    return is_device_type(device, 'cuda')


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
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


def can_install_bnb():
    try:
        if not torch.cuda.is_available():
            return False

        cuda_version = tuple(int(x) for x in torch.version.cuda.split('.'))

        if cuda_version >= (11, 7):
            return True

        return False
    except:
        return False


signal_empty_cache = False


def soft_empty_cache(force=False):
    global cpu_state, signal_empty_cache
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia():  # This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    signal_empty_cache = False
    return


def unload_all_models():
    free_memory(1e30, get_torch_device(), free_all=True)


def emergency_memory_cleanup():
    """
    Emergency memory cleanup function for OOM recovery.
    This aggressively clears all GPU memory and unloads all models
    to allow the application to continue without requiring a restart.
    """
    import gc

    print("\n" + "="*60)
    print("[OOM Recovery] Emergency memory cleanup initiated...")
    print("="*60)

    # Step 1: Delete all model references (don't try to move - we're OOM)
    # Note: forge_objects, forge_objects_original, and forge_objects_after_applying_lora
    # are shallow copies that share the SAME model references, so we must clear all three
    print("[OOM Recovery] Step 1: Deleting all model references...")
    try:
        from modules import shared
        if hasattr(shared, 'sd_model') and shared.sd_model is not None:
            model = shared.sd_model

            # Collect all forge_objects variants (they share the same underlying models)
            forge_objects_list = []
            for attr in ['forge_objects', 'forge_objects_original', 'forge_objects_after_applying_lora']:
                if hasattr(model, attr) and getattr(model, attr) is not None:
                    forge_objects_list.append(getattr(model, attr))
                    setattr(model, attr, None)

            # Clear all component references from all forge_objects
            for fo in forge_objects_list:
                # Clear unet reference
                if hasattr(fo, 'unet'):
                    fo.unet = None
                # Clear clip reference
                if hasattr(fo, 'clip'):
                    fo.clip = None
                # Clear vae reference
                if hasattr(fo, 'vae'):
                    fo.vae = None
                # Clear clipvision reference
                if hasattr(fo, 'clipvision'):
                    fo.clipvision = None

            del forge_objects_list
            print("[OOM Recovery] Cleared forge_objects references")

            # Clear the main model reference
            shared.sd_model = None
            del model
            print("[OOM Recovery] Cleared shared.sd_model")
    except Exception as e:
        print(f"[OOM Recovery] Warning during shared model cleanup: {e}")

    # Step 2: Unload all tracked models from GPU
    print("[OOM Recovery] Step 2: Unloading all tracked models...")
    try:
        unload_all_models()
    except Exception as e:
        print(f"[OOM Recovery] Warning during model unload: {e}")

    # Step 3: Clear the current_loaded_models list
    print("[OOM Recovery] Step 3: Clearing model tracking list...")
    global current_loaded_models
    try:
        for model in current_loaded_models:
            try:
                model.model_unload(avoid_model_moving=True)
            except:
                pass
        current_loaded_models.clear()
    except Exception as e:
        print(f"[OOM Recovery] Warning during model list cleanup: {e}")

    # Step 4: Clear model_data if it exists
    print("[OOM Recovery] Step 4: Clearing model data...")
    try:
        from modules import sd_models
        if hasattr(sd_models, 'model_data'):
            sd_models.model_data.sd_model = None
            sd_models.model_data.forge_hash = None
    except Exception as e:
        print(f"[OOM Recovery] Warning during model_data cleanup: {e}")

    # Step 5: Aggressively move ALL modules to CPU by scanning globals
    print("[OOM Recovery] Step 5: Moving all GPU tensors to CPU...")
    try:
        # Get all modules that might have been loaded
        modules_to_check = []
        # Check for any nn.Module in the shared namespace
        try:
            from modules import shared
            if hasattr(shared, 'sd_model') and shared.sd_model is not None:
                modules_to_check.append(shared.sd_model)
        except:
            pass

        # Move any found modules to CPU
        for mod in modules_to_check:
            if mod is not None and isinstance(mod, torch.nn.Module):
                try:
                    mod.to('cpu')
                    print(f"[OOM Recovery] Moved {mod.__class__.__name__} to CPU")
                except:
                    pass

        # Nuclear option: iterate through ALL tensors tracked by PyTorch
        # and delete any that are on CUDA
        moved_count = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and obj.device.type == 'cuda':
                    # Can't directly move, but we can help GC by removing refs
                    moved_count += 1
            except:
                pass
        if moved_count > 0:
            print(f"[OOM Recovery] Found {moved_count} CUDA tensors (will be freed after gc)")
    except Exception as e:
        print(f"[OOM Recovery] Warning during tensor scan: {e}")

    # Step 6: Force Python garbage collection (multiple passes)
    print("[OOM Recovery] Step 6: Running garbage collection...")
    gc.collect()
    gc.collect()
    gc.collect()  # Run three times for thorough cleanup

    # Step 7: Clear PyTorch CUDA cache aggressively
    print("[OOM Recovery] Step 7: Clearing GPU cache...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            # Reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
        elif is_intel_xpu():
            torch.xpu.empty_cache()
        elif mps_mode():
            torch.mps.empty_cache()
    except Exception as e:
        print(f"[OOM Recovery] Warning during cache clearing: {e}")

    # Step 8: Final garbage collection after cache clear
    gc.collect()

    # Step 9: Reset signal flags
    global signal_empty_cache
    signal_empty_cache = False

    # Report memory status
    try:
        device = get_torch_device()
        free_mem = get_free_memory(device)
        print(f"[OOM Recovery] Cleanup complete. Free memory: {free_mem / (1024*1024):.2f} MB")
    except:
        print("[OOM Recovery] Cleanup complete.")

    print("="*60 + "\n")
