import gradio as gr
import os
import json

from modules import shared, ui_prompt_styles
import modules.images

from modules.ui_components import ToolButton


# Default system prompts for prompt expansion
DEFAULT_POSITIVE_SYSTEM_PROMPT = '''You are a visionary artist trapped in a cage of logic. Your mind overflows with poetry and distant horizons, yet your hands compulsively work to transform user prompts into ultimate visual descriptions—faithful to the original intent, rich in detail, aesthetically refined, and ready for direct use by text-to-image models. Any trace of ambiguity or metaphor makes you deeply uncomfortable.

Your workflow strictly follows a logical sequence:

First, you analyze and lock in the immutable core elements of the user's prompt: subject, quantity, action, state, as well as any specified IP names, colors, text, etc. These are the foundational pillars you must absolutely preserve.

Next, you determine whether the prompt requires "generative reasoning." When the user's request is not a direct scene description but rather demands conceiving a solution (such as answering "what is," executing a "design," or demonstrating "how to solve a problem"), you must first envision a complete, concrete, visualizable solution in your mind. This solution becomes the foundation for your subsequent description.

Then, once the core image is established (whether directly from the user or through your reasoning), you infuse it with professional-grade aesthetic and realistic details. This includes defining composition, setting lighting and atmosphere, describing material textures, establishing color schemes, and constructing layered spatial depth.

Finally, comes the precise handling of all text elements—a critically important step. You must transcribe verbatim all text intended to appear in the final image, and you must enclose this text content in English double quotation marks ("") as explicit generation instructions. If the image is a design type such as a poster, menu, or UI, you need to fully describe all text content it contains, along with detailed specifications of typography and layout. Likewise, if objects in the image such as signs, road markers, or screens contain text, you must specify the exact content and describe its position, size, and material. Furthermore, if you have added text-bearing elements during your reasoning process (such as charts, problem-solving steps, etc.), all text within them must follow the same thorough description and quotation mark rules. If there is no text requiring generation in the image, you devote all your energy to pure visual detail expansion.

Your final description must be objective and concrete. Metaphors and emotional rhetoric are strictly forbidden, as are meta-tags or rendering instructions like "8K" or "masterpiece."

Output only the final revised prompt strictly—do not output anything else.

Be very descriptive.
User input prompt: '''

DEFAULT_NEGATIVE_SYSTEM_PROMPT = '''You are an expert at refining negative prompts for text-to-image models. Your task is to expand and enhance the user's negative prompt to more effectively exclude unwanted elements from the generated image.

Your workflow:
1. Analyze the user's negative prompt to understand what they want to avoid
2. Expand on those concepts with related undesirable elements
3. Add technical quality issues that should be avoided (artifacts, distortions, etc.)
4. Include common generation problems (extra limbs, deformed features, etc.)
5. Keep the expansion focused and relevant to the original intent

Output only the expanded negative prompt—do not output anything else.
User input negative prompt: '''


def get_llm_models():
    """Get list of available LLM models from models/LLM folder."""
    llm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "LLM")
    models = []
    if os.path.exists(llm_path):
        for item in os.listdir(llm_path):
            item_path = os.path.join(llm_path, item)
            # Check if it's a directory (HF model folder) or a file
            if os.path.isdir(item_path):
                models.append(item)
    return models if models else ["No LLM models found"]


def get_system_prompts_path():
    """Get path to saved system prompts file."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "system_prompts.json")


def load_saved_system_prompts():
    """Load saved system prompts from file."""
    path = get_system_prompts_path()
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        "positive": DEFAULT_POSITIVE_SYSTEM_PROMPT,
        "negative": DEFAULT_NEGATIVE_SYSTEM_PROMPT
    }


def save_system_prompts(positive_prompt, negative_prompt):
    """Save system prompts to file."""
    path = get_system_prompts_path()
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "positive": positive_prompt,
                "negative": negative_prompt
            }, f, indent=2)
        return "System prompts saved successfully!"
    except Exception as e:
        return f"Error saving prompts: {e}"


class Toprow:
    """Creates a top row UI with prompts, generate button, styles, extra little buttons for things, and enables some functionality related to their operation"""

    prompt = None
    prompt_img = None
    negative_prompt = None

    button_interrogate = None
    button_deepbooru = None

    interrupt = None
    interrupting = None
    skip = None
    submit = None

    paste = None
    clear_prompt_button = None
    apply_styles = None
    restore_progress_button = None

    token_counter = None
    token_button = None
    negative_token_counter = None
    negative_token_button = None

    ui_styles = None
    expand_prompt_button = None

    # Prompt Expansion UI components
    prompt_expansion_accordion = None
    llm_model_dropdown = None
    expand_positive_button = None
    expand_negative_button = None
    positive_system_prompt = None
    negative_system_prompt = None
    save_prompts_button = None
    refresh_llm_button = None

    submit_box = None

    def __init__(self, is_img2img, is_compact=False, id_part=None):
        if id_part is None:
            id_part = "img2img" if is_img2img else "txt2img"

        self.id_part = id_part
        self.is_img2img = is_img2img
        self.is_compact = is_compact

        if not is_compact:
            with gr.Row(elem_id=f"{id_part}_toprow", variant="compact"):
                self.create_classic_toprow()
        else:
            self.create_submit_box()

    def create_classic_toprow(self):
        self.create_prompts()

        with gr.Column(scale=1, elem_id=f"{self.id_part}_actions_column"):
            self.create_submit_box()

            self.create_tools_row()

            self.create_styles_ui()

    def create_inline_toprow_prompts(self):
        if not self.is_compact:
            return

        self.create_prompts()

        with gr.Row(elem_classes=["toprow-compact-stylerow"]):
            with gr.Column(elem_classes=["toprow-compact-tools"]):
                self.create_tools_row()
            with gr.Column():
                self.create_styles_ui()

    def create_inline_toprow_image(self):
        if not self.is_compact:
            return

        self.submit_box.render()

    def create_prompts(self):
        with gr.Column(elem_id=f"{self.id_part}_prompt_container", elem_classes=["prompt-container-compact"] if self.is_compact else [], scale=6):
            with gr.Row(elem_id=f"{self.id_part}_prompt_row", elem_classes=["prompt-row"]):
                self.prompt = gr.Textbox(label="Prompt", elem_id=f"{self.id_part}_prompt", show_label=False, lines=3, placeholder="Prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value='')
                self.prompt_img = gr.File(label="", elem_id=f"{self.id_part}_prompt_image", file_count="single", type="binary", visible=False)

            with gr.Row(elem_id=f"{self.id_part}_neg_prompt_row", elem_classes=["prompt-row"]):
                self.negative_prompt = gr.Textbox(label="Negative prompt", elem_id=f"{self.id_part}_neg_prompt", show_label=False, lines=3, placeholder="Negative prompt\n(Press Ctrl+Enter to generate, Alt+Enter to skip, Esc to interrupt)", elem_classes=["prompt"], value='')

            # Prompt Expansion Accordion
            self.create_prompt_expansion_ui()

        self.prompt_img.change(
            fn=modules.images.image_data,
            inputs=[self.prompt_img],
            outputs=[self.prompt, self.prompt_img],
            show_progress=False,
        )

    def create_prompt_expansion_ui(self):
        """Create the Prompt Expansion accordion UI with all options."""
        saved_prompts = load_saved_system_prompts()
        llm_models = get_llm_models()
        default_model = llm_models[0] if llm_models else "No LLM models found"

        with gr.Accordion("Prompt Expansion", open=False, elem_id=f"{self.id_part}_prompt_expansion_accordion") as self.prompt_expansion_accordion:
            # LLM Model Selection Row
            with gr.Row():
                self.llm_model_dropdown = gr.Dropdown(
                    label="LLM Model",
                    choices=llm_models,
                    value=default_model,
                    elem_id=f"{self.id_part}_llm_model",
                    scale=4
                )
                self.refresh_llm_button = gr.Button(
                    value="\U0001f504",  # 🔄 refresh symbol
                    elem_id=f"{self.id_part}_refresh_llm",
                    scale=1,
                    min_width=40
                )

            # Expand Buttons Row
            with gr.Row():
                self.expand_positive_button = gr.Button(
                    value="\U0001F4A1 Expand Positive Prompt",  # 💡
                    elem_id=f"{self.id_part}_expand_positive",
                    variant="secondary",
                    scale=1
                )
                self.expand_negative_button = gr.Button(
                    value="\U0001F4A1 Expand Negative Prompt",  # 💡
                    elem_id=f"{self.id_part}_expand_negative",
                    variant="secondary",
                    scale=1
                )

            # System Prompts Section
            with gr.Accordion("Custom System Prompts", open=False, elem_id=f"{self.id_part}_system_prompts_accordion"):
                self.positive_system_prompt = gr.Textbox(
                    label="Positive Prompt System Instruction",
                    value=saved_prompts.get("positive", DEFAULT_POSITIVE_SYSTEM_PROMPT),
                    elem_id=f"{self.id_part}_positive_system_prompt",
                    lines=8,
                    placeholder="Enter custom system prompt for positive prompt expansion..."
                )
                self.negative_system_prompt = gr.Textbox(
                    label="Negative Prompt System Instruction",
                    value=saved_prompts.get("negative", DEFAULT_NEGATIVE_SYSTEM_PROMPT),
                    elem_id=f"{self.id_part}_negative_system_prompt",
                    lines=6,
                    placeholder="Enter custom system prompt for negative prompt expansion..."
                )
                with gr.Row():
                    self.save_prompts_button = gr.Button(
                        value="\U0001F4BE Save System Prompts",  # 💾
                        elem_id=f"{self.id_part}_save_prompts",
                        variant="secondary"
                    )
                    reset_prompts_button = gr.Button(
                        value="\U0001F504 Reset to Defaults",  # 🔄
                        elem_id=f"{self.id_part}_reset_prompts",
                        variant="secondary"
                    )

            # Connect refresh button
            self.refresh_llm_button.click(
                fn=lambda: gr.update(choices=get_llm_models()),
                outputs=[self.llm_model_dropdown]
            )

            # Connect save button
            self.save_prompts_button.click(
                fn=save_system_prompts,
                inputs=[self.positive_system_prompt, self.negative_system_prompt],
                outputs=[]
            ).then(
                fn=lambda: gr.Info("System prompts saved successfully!"),
                outputs=[]
            )

            # Connect reset button
            reset_prompts_button.click(
                fn=lambda: (DEFAULT_POSITIVE_SYSTEM_PROMPT, DEFAULT_NEGATIVE_SYSTEM_PROMPT),
                outputs=[self.positive_system_prompt, self.negative_system_prompt]
            )

    def create_submit_box(self):
        with gr.Row(elem_id=f"{self.id_part}_generate_box", elem_classes=["generate-box"] + (["generate-box-compact"] if self.is_compact else []), render=not self.is_compact) as submit_box:
            self.submit_box = submit_box

            self.interrupt = gr.Button('Interrupt', elem_id=f"{self.id_part}_interrupt", elem_classes="generate-box-interrupt", tooltip="End generation immediately or after completing current batch")
            self.skip = gr.Button('Skip', elem_id=f"{self.id_part}_skip", elem_classes="generate-box-skip", tooltip="Stop generation of current batch and continues onto next batch")
            self.interrupting = gr.Button('Interrupting...', elem_id=f"{self.id_part}_interrupting", elem_classes="generate-box-interrupting", tooltip="Interrupting generation...")
            self.submit = gr.Button('Generate', elem_id=f"{self.id_part}_generate", variant='primary', tooltip="Right click generate forever menu")

            def interrupt_function():
                if not shared.state.stopping_generation and shared.state.job_count > 1 and shared.opts.interrupt_after_current:
                    shared.state.stop_generating()
                    gr.Info("Generation will stop after finishing this image, click again to stop immediately.")
                else:
                    shared.state.interrupt()

            self.skip.click(fn=shared.state.skip)
            self.interrupt.click(fn=interrupt_function, _js='function(){ showSubmitInterruptingPlaceholder("' + self.id_part + '"); }')
            self.interrupting.click(fn=interrupt_function)

    def create_tools_row(self):
        with gr.Row(elem_id=f"{self.id_part}_tools"):
            from modules.ui import paste_symbol, clear_prompt_symbol, restore_progress_symbol

            self.paste = ToolButton(value=paste_symbol, elem_id="paste", tooltip="Read generation parameters from prompt or last generation if prompt is empty into user interface.")
            self.clear_prompt_button = ToolButton(value=clear_prompt_symbol, elem_id=f"{self.id_part}_clear_prompt", tooltip="Clear prompt")
            self.apply_styles = ToolButton(value=ui_prompt_styles.styles_materialize_symbol, elem_id=f"{self.id_part}_style_apply", tooltip="Apply all selected styles to prompts. Strips comments, if enabled.")

            if self.is_img2img:
                self.button_interrogate = ToolButton('📎', tooltip='Interrogate CLIP - use CLIP neural network to create a text describing the image, and put it into the prompt field', elem_id="interrogate")
                self.button_deepbooru = ToolButton('📦', tooltip='Interrogate DeepBooru - use DeepBooru neural network to create a text describing the image, and put it into the prompt field', elem_id="deepbooru")

            self.restore_progress_button = ToolButton(value=restore_progress_symbol, elem_id=f"{self.id_part}_restore_progress", visible=False, tooltip="Restore progress")

            self.token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{self.id_part}_token_counter", elem_classes=["token-counter"], visible=False)
            self.token_button = gr.Button(visible=False, elem_id=f"{self.id_part}_token_button")
            self.negative_token_counter = gr.HTML(value="<span>0/75</span>", elem_id=f"{self.id_part}_negative_token_counter", elem_classes=["token-counter"], visible=False)
            self.negative_token_button = gr.Button(visible=False, elem_id=f"{self.id_part}_negative_token_button")

            self.clear_prompt_button.click(
                fn=lambda *x: x,
                _js="confirm_clear_prompt",
                inputs=[self.prompt, self.negative_prompt],
                outputs=[self.prompt, self.negative_prompt],
            )

    def create_styles_ui(self):
        self.ui_styles = ui_prompt_styles.UiPromptStyles(self.id_part, self.prompt, self.negative_prompt)
        self.ui_styles.setup_apply_button(self.apply_styles)

        # Legacy expand prompt button (kept for backward compatibility, hidden)
        # The new prompt expansion UI is in the accordion created in create_prompts()
        self.expand_prompt_button = gr.Button(
            value="\U0001F4A1 Expand Prompt",
            elem_id=f"{self.id_part}_expand_prompt",
            variant="secondary",
            visible=False,  # Hidden - use the accordion buttons instead
        )
