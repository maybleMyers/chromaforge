import gradio as gr

from modules import scripts
from backend import memory_management


class NeverOOMForForge(scripts.Script):
    sorting_priority = 18

    def __init__(self):
        self.previous_unet_enabled = False
        self.original_vram_state = memory_management.vram_state
        self.original_blocks_to_swap = 0

    def title(self):
        return "Never OOM Integrated"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            unet_enabled = gr.Checkbox(label='Enabled for UNet (always maximize offload)', value=False)
            vae_enabled = gr.Checkbox(label='Enabled for VAE (always tiled)', value=False)
        return unet_enabled, vae_enabled

    def process(self, p, *script_args, **kwargs):
        unet_enabled, vae_enabled = script_args

        if unet_enabled:
            print('NeverOOM Enabled for UNet (always maximize offload)')

        if vae_enabled:
            print('NeverOOM Enabled for VAE (always tiled)')

        memory_management.VAE_ALWAYS_TILED = vae_enabled

        if self.previous_unet_enabled != unet_enabled:
            memory_management.unload_all_models()
            if unet_enabled:
                # Store original states
                self.original_vram_state = memory_management.vram_state
                self.original_blocks_to_swap = memory_management.get_blocks_to_swap()

                # Enable maximum block swapping (equivalent to NO_VRAM)
                try:
                    from modules import shared
                    if hasattr(shared, 'opts'):
                        shared.opts.set('blocks_to_swap', 30)  # Maximum swapping
                except:
                    pass

                memory_management.vram_state = memory_management.VRAMState.NO_VRAM
                print('Never OOM: Enabled maximum block swapping (30 blocks)')
            else:
                # Restore original states
                memory_management.vram_state = self.original_vram_state
                try:
                    from modules import shared
                    if hasattr(shared, 'opts'):
                        shared.opts.set('blocks_to_swap', self.original_blocks_to_swap)
                except:
                    pass
                print(f'Never OOM: Restored original block swapping ({self.original_blocks_to_swap} blocks)')

            print(f'VRAM State Changed To {memory_management.vram_state.name}')
            self.previous_unet_enabled = unet_enabled

        return
