"""
Chroma Radiance Controls Script
Adds conditional radiance-specific controls to the Chroma interface following proper Forge patterns
"""

import gradio as gr
import modules.scripts as scripts
from modules import shared, script_callbacks, processing
from modules.ui_components import FormRow
import modules.ui_settings as ui_settings

class ChromaRadianceControls(scripts.Script):
    """Script to add conditional radiance controls to Chroma interface."""
    
    def __init__(self):
        self.radiance_components = {}
        self.model_type_component = None
        
    def title(self):
        return "Chroma Radiance Controls"
        
    def show(self, is_img2img):
        return scripts.AlwaysVisible
        
    def ui(self, is_img2img):
        """Create radiance-specific UI components following Forge patterns."""
        tab_name = "img2img" if is_img2img else "txt2img"
        
        # Create the interface within proper Gradio context
        with gr.Blocks() as interface:
            # Create model type selector (this will work with extra-options system)
            with gr.Group() as model_type_group:
                model_type = gr.Radio(
                    label="Chroma Model Type",
                    choices=["Standard", "Radiance"], 
                    value=lambda: shared.opts.chroma_model_type,
                    elem_id=f"chroma_model_type_{tab_name}"
                )
            
            # Create radiance controls (initially hidden)
            with gr.Group(visible=False, elem_id=f"chroma_radiance_group_{tab_name}") as radiance_group:
                gr.HTML("<h4>Radiance Settings</h4>")
                with FormRow():
                    radiance_guidance = gr.Slider(
                        minimum=0.0, maximum=10.0, step=0.1,
                        value=lambda: shared.opts.chroma_radiance_guidance,
                        label="Radiance Guidance",
                        elem_id=f"chroma_radiance_guidance_{tab_name}"
                    )
                    radiance_attn_padding = gr.Slider(
                        minimum=1, maximum=16, step=1,
                        value=lambda: shared.opts.chroma_radiance_attn_padding,
                        label="Radiance Attention Padding",
                        elem_id=f"chroma_radiance_attn_padding_{tab_name}"
                    )
            
            # Set up conditional visibility within Gradio context
            def on_model_type_change(model_type_value):
                """Handle model type changes to show/hide radiance controls."""
                is_radiance = (model_type_value == "Radiance")
                
                # Update the shared setting
                try:
                    shared.opts.chroma_model_type = model_type_value
                except:
                    pass  # Ignore if setting doesn't exist yet
                    
                return gr.update(visible=is_radiance)
            
            # Set up the event handler within Gradio context
            model_type.change(
                fn=on_model_type_change,
                inputs=[model_type],
                outputs=[radiance_group],
                show_progress=False
            )
            
        # Store components for processing
        self.radiance_components[tab_name] = {
            'model_type': model_type,
            'group': radiance_group,
            'guidance': radiance_guidance,
            'attn_padding': radiance_attn_padding
        }
        
        # Return all components for processing integration
        return [model_type, radiance_guidance, radiance_attn_padding]
    
    def process(self, p, *args):
        """Process function called during generation."""
        if len(args) >= 3:
            model_type = args[0] if args[0] is not None else "Standard"
            guidance = args[1] if args[1] is not None else 0.0
            attn_padding = args[2] if args[2] is not None else 1
            
            # Only apply radiance processing if model type is set to Radiance
            if model_type == "Radiance":
                # Add radiance parameters to processing
                setattr(p, 'radiance_guidance', float(guidance))
                setattr(p, 'radiance_attn_padding', int(attn_padding))
                
                # Mark this as a radiance processing job
                setattr(p, 'use_radiance_processing', True)
                
                # Add to extra generation params for info text
                if not hasattr(p, 'extra_generation_params'):
                    p.extra_generation_params = {}
                
                p.extra_generation_params.update({
                    "Radiance Mode": True,
                    "Radiance Guidance": float(guidance),
                    "Radiance Attention Padding": int(attn_padding),
                })
            else:
                # Ensure radiance processing is disabled for standard mode
                setattr(p, 'use_radiance_processing', False)


def ensure_processing_integration():
    """Ensure radiance processing integration is available."""
    try:
        # Import radiance utilities and processing
        from modules.radiance_model_utils import is_radiance_model
        from modules.processing_radiance import process_images_radiance
        
        # Monkey patch the main processing function to handle radiance
        if not hasattr(processing, '_original_process_images'):
            processing._original_process_images = processing.process_images
        
        def enhanced_process_images(p):
            """Enhanced processing function that handles radiance."""
            # Check if this should use radiance processing
            should_use_radiance = (
                hasattr(p, 'use_radiance_processing') and 
                p.use_radiance_processing and 
                is_radiance_model(shared.sd_model)
            )
            
            if should_use_radiance:
                try:
                    print("Using radiance processing")
                    
                    # Import radiance processing class
                    from modules.processing_radiance import RadianceProcessing, process_images_radiance
                    
                    # Create radiance processing object if not already one
                    if not isinstance(p, RadianceProcessing):
                        # Transfer all attributes to RadianceProcessing object
                        radiance_kwargs = {}
                        for attr_name in dir(p):
                            if not attr_name.startswith('_') and hasattr(p, attr_name):
                                attr_value = getattr(p, attr_name)
                                if not callable(attr_value):
                                    radiance_kwargs[attr_name] = attr_value
                        
                        # Create new RadianceProcessing instance
                        radiance_p = RadianceProcessing(**radiance_kwargs)
                        
                        # Copy any additional attributes that might have been added
                        for attr in ['radiance_guidance', 'radiance_attn_padding', 'use_radiance_processing']:
                            if hasattr(p, attr):
                                setattr(radiance_p, attr, getattr(p, attr))
                        
                        return process_images_radiance(radiance_p)
                    else:
                        return process_images_radiance(p)
                        
                except Exception as e:
                    print(f"Radiance processing failed, falling back to standard: {e}")
                    # Fall back to standard processing
                    return processing._original_process_images(p)
            else:
                # Use standard processing
                return processing._original_process_images(p)
        
        # Replace the processing function
        processing.process_images = enhanced_process_images
        print("Chroma Radiance processing integration enabled")
        
    except ImportError as e:
        print(f"Note: Radiance processing components not available: {e}")


def setup_model_change_detection():
    """Setup automatic model type detection when models are loaded."""
    try:
        from modules.radiance_model_utils import auto_detect_and_set_model_type
        
        # Auto-detect on current model if available
        if shared.sd_model is not None:
            auto_detect_and_set_model_type()
            
    except ImportError:
        print("Note: Model detection utilities not available")


def initialize():
    """Initialize the radiance controls system."""
    ensure_processing_integration()
    setup_model_change_detection()


# Initialize when the script loads
script_callbacks.on_before_ui(initialize)
print("Chroma Radiance Controls script loaded successfully")