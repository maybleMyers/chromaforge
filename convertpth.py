import torch
from safetensors.torch import save_file

def convert_pth_to_safetensors(pth_file: str, safetensors_file: str):
    """
    Converts a PyTorch .pth file to a .safetensors file.

    Args:
        pth_file (str): Path to the .pth file to convert.
        safetensors_file (str): Path to save the .safetensors file.

    Returns:
        None
    """
    # Load the .pth file
    state_dict = torch.load(pth_file, map_location="cpu")

    # Ensure the state_dict is a dictionary
    if not isinstance(state_dict, dict):
        raise ValueError("The .pth file must contain a dictionary-like object.")

    # Save the state_dict as a .safetensors file
    save_file(state_dict, safetensors_file)
    print(f"Converted {pth_file} to {safetensors_file}")

convert_pth_to_safetensors("source.pth", "output.safetensors")