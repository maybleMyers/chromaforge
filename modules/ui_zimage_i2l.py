"""
Z-Image Image-to-LoRA (i2L) UI

Provides a dedicated tab for generating LoRA weights from style reference images.
The generated LoRA can be saved and used for style transfer during image generation.
"""

import gradio as gr
import os
from modules import shared
from modules.paths_internal import models_path
from modules.call_queue import wrap_gradio_gpu_call
from modules.ui_components import FormRow


def create_i2l_interface():
    """Create the Image-to-LoRA interface."""

    with gr.Blocks(analytics_enabled=False) as i2l_interface:
        gr.Markdown("""
## Z-Image Image-to-LoRA (i2L)

Upload style reference images to generate a LoRA that captures their style.
The generated LoRA can then be used with Z-Image models for style transfer.

**Requirements:** Models in `models/Z-Image-i2L/` (see README for download instructions)
        """)

        with FormRow():
            with gr.Column(scale=2):
                gr.HTML("<p class='text-gray-500' style='margin-bottom: 0.3em;'>Style Reference Images (up to 6)</p>")
                with gr.Row():
                    style_img1 = gr.Image(label="Style 1", source="upload", interactive=True, type="pil", elem_id="i2l_style_1", height=150)
                    style_img2 = gr.Image(label="Style 2", source="upload", interactive=True, type="pil", elem_id="i2l_style_2", height=150)
                    style_img3 = gr.Image(label="Style 3", source="upload", interactive=True, type="pil", elem_id="i2l_style_3", height=150)
                with gr.Row():
                    style_img4 = gr.Image(label="Style 4", source="upload", interactive=True, type="pil", elem_id="i2l_style_4", height=150)
                    style_img5 = gr.Image(label="Style 5", source="upload", interactive=True, type="pil", elem_id="i2l_style_5", height=150)
                    style_img6 = gr.Image(label="Style 6", source="upload", interactive=True, type="pil", elem_id="i2l_style_6", height=150)

            with gr.Column(scale=1):
                lora_name = gr.Textbox(
                    label="LoRA Name",
                    value="generated_style",
                    placeholder="Enter a name for the generated LoRA",
                    elem_id="i2l_lora_name",
                )

                generate_btn = gr.Button(
                    "Generate LoRA",
                    variant="primary",
                    elem_id="i2l_generate_btn",
                )

                clear_btn = gr.Button("Clear All Images", elem_id="i2l_clear_btn")

        with FormRow():
            status_output = gr.Textbox(
                label="Status",
                lines=5,
                max_lines=10,
                interactive=False,
                elem_id="i2l_status",
            )

        with FormRow():
            output_path = gr.Textbox(
                label="Generated LoRA Path",
                interactive=False,
                elem_id="i2l_output_path",
            )

        # Collect all style image components
        style_images = [style_img1, style_img2, style_img3, style_img4, style_img5, style_img6]

        def clear_images():
            """Clear all images."""
            return [None] * 6

        def generate_lora(img1, img2, img3, img4, img5, img6, name):
            """Generate LoRA from style images."""
            # Collect non-None images
            all_imgs = [img1, img2, img3, img4, img5, img6]
            pil_images = [img for img in all_imgs if img is not None]

            if len(pil_images) == 0:
                return "Error: Please upload at least one style image.", ""

            if not name:
                name = "generated_style"

            # Clean the name for filesystem
            name = "".join(c for c in name if c.isalnum() or c in "._- ")
            if not name:
                name = "generated_style"

            try:
                from backend.diffusion_engine.zimage import ZImage

                status_msgs = []
                status_msgs.append(f"Processing {len(pil_images)} style images...")

                # Get models directory
                models_dir = os.path.join(models_path, "Z-Image-i2L")
                i2l_model_path = os.path.join(models_dir, "model.safetensors")

                # Check if models exist
                if not os.path.exists(i2l_model_path):
                    return f"Error: Image2LoRA model not found at {i2l_model_path}", ""

                siglip_path = os.path.join(models_dir, "SigLIP2-G384", "model.safetensors")
                dinov3_path = os.path.join(models_dir, "DINOv3-7B", "model.safetensors")

                if not os.path.exists(siglip_path):
                    return f"Error: SigLIP2 encoder not found at {siglip_path}", ""
                if not os.path.exists(dinov3_path):
                    return f"Error: DINOv3 encoder not found at {dinov3_path}", ""

                status_msgs.append("Loading encoders and generating LoRA...")

                # Generate LoRA
                lora_dict = ZImage.generate_lora_from_images(
                    images=pil_images,
                    models_dir=models_dir,
                    i2l_model_path=i2l_model_path,
                )

                status_msgs.append(f"Generated LoRA with {len(lora_dict)} weight keys")

                # Save LoRA
                output_dir = os.path.join(models_path, "Lora")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{name}.safetensors")

                ZImage.save_lora_to_file(lora_dict, output_file)
                status_msgs.append(f"Saved LoRA to {output_file}")

                return "\n".join(status_msgs), output_file

            except Exception as e:
                import traceback
                error_msg = f"Error generating LoRA: {str(e)}\n\n{traceback.format_exc()}"
                return error_msg, ""

        clear_btn.click(
            fn=clear_images,
            outputs=style_images,
        )

        generate_btn.click(
            fn=wrap_gradio_gpu_call(generate_lora),
            inputs=style_images + [lora_name],
            outputs=[status_output, output_path],
        )

    return i2l_interface


# Register the tab
def register_i2l_tab():
    """Register the i2L tab with the WebUI."""
    from modules import ui

    # Create and return the interface
    interface = create_i2l_interface()
    return interface, "Z-Image i2L", "zimage_i2l"


# Add to tab names if not present
if "Z-Image i2L" not in shared.tab_names:
    shared.tab_names.append("Z-Image i2L")
