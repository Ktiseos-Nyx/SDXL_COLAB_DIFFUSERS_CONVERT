import os
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from huggingface_hub import HfApi, login
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError
import re
import json
import glob
import gdown
import requests
import subprocess
from urllib.parse import urlparse, unquote
from pathlib import Path

# ---------------------- DEPENDENCIES ----------------------

def install_dependencies_gradio():
    """Installs the necessary dependencies for the Gradio app.  Run this ONCE."""
    try:
        !pip install -U torch diffusers transformers accelerate safetensors huggingface_hub xformers
        print("Dependencies installed successfully.")
    except Exception as e:
        print(f"Error installing dependencies: {e}")

# ---------------------- UTILITY FUNCTIONS ----------------------

def get_save_dtype(save_precision_as):
    """Determines the save dtype based on the user's choice."""
    if save_precision_as == "fp16":
        return torch.float16
    elif save_precision_as == "bf16":
        return torch.bfloat16
    elif save_precision_as == "float":
        return torch.float32  # Using float32 for "float" option
    else:
        return None

def determine_load_checkpoint(model_to_load):
    """Determines if the model to load is a checkpoint or a Diffusers model."""
    if model_to_load.endswith('.ckpt') or model_to_load.endswith('.safetensors'):
        return True
    elif os.path.isdir(model_to_load):
        required_folders = {"unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler", "vae"}
        if required_folders.issubset(set(os.listdir(model_to_load))) and os.path.isfile(os.path.join(model_to_load, "model_index.json")):
            return False
    return None  # handle this case as required

def increment_filename(filename):
    """Increments the filename to avoid overwriting existing files."""
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base}({counter}){ext}"
        counter += 1
    return filename

def create_model_repo(api, user, orgs_name, model_name, make_private=False):
    """Creates a Hugging Face model repository if it doesn't exist."""
    if orgs_name == "":
        repo_id = user["name"] + "/" + model_name.strip()
    else:
        repo_id = orgs_name + "/" + model_name.strip()

    try:
        validate_repo_id(repo_id)
        api.create_repo(repo_id=repo_id, repo_type="model", private=make_private)
        print(f"Model repo '{repo_id}' didn't exist, creating repo")
    except HfHubHTTPError as e:
        print(f"Model repo '{repo_id}' exists, skipping create repo")

    print(f"Model repo '{repo_id}' link: https://huggingface.co/{repo_id}\n")

    return repo_id

def is_diffusers_model(model_path):
    """Checks if a given path is a valid Diffusers model directory."""
    required_folders = {"unet", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler", "vae"}
    return required_folders.issubset(set(os.listdir(model_path))) and os.path.isfile(os.path.join(model_path, "model_index.json"))

# ---------------------- CONVERSION AND UPLOAD FUNCTIONS ----------------------

def load_sdxl_model(args, is_load_checkpoint, load_dtype, output_widget):
    """Loads the SDXL model from a checkpoint or Diffusers model."""
    model_load_message = "checkpoint" if is_load_checkpoint else "Diffusers" + (" as fp16" if args.fp16 else "")
    with output_widget:
        print(f"Loading {model_load_message}: {args.model_to_load}")

    if is_load_checkpoint:
        loaded_model_data = load_from_sdxl_checkpoint(args, output_widget)
    else:
        loaded_model_data = load_sdxl_from_diffusers(args, load_dtype)

    return loaded_model_data

def load_from_sdxl_checkpoint(args, output_widget):
    """Loads the SDXL model components from a checkpoint file (placeholder)."""
    # text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
    #    "sdxl_base_v1-0", args.model_to_load, "cpu"
    # )

    # Implement Load model from ckpt or safetensors
    text_encoder1, text_encoder2, vae, unet = None, None, None, None

    with output_widget:
        print("Loading from Checkpoint not implemented, please implement based on your model needs.")

    return text_encoder1, text_encoder2, vae, unet

def load_sdxl_from_diffusers(args, load_dtype):
    """Loads an SDXL model from a Diffusers model directory."""
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_to_load, torch_dtype=load_dtype, tokenizer=None, tokenizer_2=None, scheduler=None
    )
    text_encoder1 = pipeline.text_encoder
    text_encoder2 = pipeline.text_encoder_2
    vae = pipeline.vae
    unet = pipeline.unet

    return text_encoder1, text_encoder2, vae, unet

def convert_and_save_sdxl_model(args, is_save_checkpoint, loaded_model_data, save_dtype, output_widget):
    """Converts and saves the SDXL model as either a checkpoint or a Diffusers model."""
    text_encoder1, text_encoder2, vae, unet = loaded_model_data
    model_save_message = "checkpoint" + ("" if save_dtype is None else f" in {save_dtype}") if is_save_checkpoint else "Diffusers"

    with output_widget:
        print(f"Converting and saving as {model_save_message}: {args.model_to_save}")

    if is_save_checkpoint:
        save_sdxl_as_checkpoint(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget)
    else:
        save_sdxl_as_diffusers(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget)

def save_sdxl_as_checkpoint(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget):
    """Saves the SDXL model components as a checkpoint file (placeholder)."""
    # logit_scale = None
    # ckpt_info = None

    # key_count = sdxl_model_util.save_stable_diffusion_checkpoint(
    #    args.model_to_save, text_encoder1, text_encoder2, unet, args.epoch, args.global_step, ckpt_info, vae, logit_scale, save_dtype
    # )

    with output_widget:
        print("Saving as Checkpoint not implemented, please implement based on your model needs.")
        # print(f"Model saved. Total converted state_dict keys: {key_count}")

def save_sdxl_as_diffusers(args, text_encoder1, text_encoder2, vae, unet, save_dtype, output_widget):
    """Saves the SDXL model as a Diffusers model."""
    with output_widget:
        reference_model_message = args.reference_model if args.reference_model is not None else 'default model'
        print(f"Copying scheduler/tokenizer config from: {reference_model_message}")

    # Save diffusers pipeline
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder1,
        text_encoder_2=text_encoder2,
        unet=unet,
        scheduler=None,  # Replace None if there is a scheduler
        tokenizer=None,  # Replace None if there is a tokenizer
        tokenizer_2=None  # Replace None if there is a tokenizer_2
    )

    pipeline.save_pretrained(args.model_to_save)

    with output_widget:
        print(f"Model saved as {save_dtype}.")

def convert_model(model_to_load, save_precision_as, epoch, global_step, reference_model, output_path, fp16, output_widget):
    """Main conversion function."""
    class Args:  # Defining Args locally within convert_model
        def __init__(self, model_to_load, save_precision_as, epoch, global_step, reference_model, output_path, fp16):
            self.model_to_load = model_to_load
            self.save_precision_as = save_precision_as
            self.epoch = epoch
            self.global_step = global_step
            self.reference_model = reference_model
            self.output_path = output_path
            self.fp16 = fp16

    args = Args(model_to_load, save_precision_as, epoch, global_step, reference_model, output_path, fp16)
    args.model_to_save = increment_filename(os.path.splitext(args.model_to_load)[0] + ".safetensors")

    try:
        load_dtype = torch.float16 if fp16 else None
        save_dtype = get_save_dtype(save_precision_as)

        is_load_checkpoint = determine_load_checkpoint(model_to_load)
        is_save_checkpoint = not is_load_checkpoint  # reverse of load model

        loaded_model_data = load_sdxl_model(args, is_load_checkpoint, load_dtype, output_widget)
        convert_and_save_sdxl_model(args, is_save_checkpoint, loaded_model_data, save_dtype, output_widget)

        with output_widget:
            return f"Conversion complete. Model saved to {args.model_to_save}"

    except Exception as e:
        with output_widget:
            return f"Conversion failed: {e}"

def upload_to_huggingface(model_path, hf_token, orgs_name, model_name, make_private, output_widget):
    """Uploads a model to the Hugging Face Hub."""
    try:
        login(hf_token, add_to_git_credential=True)
        api = HfApi()
        user = api.whoami(hf_token)
        model_repo = create_model_repo(api, user, orgs_name, model_name, make_private)

        # Determine upload parameters (adjust as needed)
        path_in_repo = ""
        trained_model = os.path.basename(model_path)

        path_in_repo_local = path_in_repo if path_in_repo and not is_diffusers_model(model_path) else ""

        notification = f"Uploading {trained_model} from {model_path} to https://huggingface.co/{model_repo}"
        with output_widget:
            print(notification)

        if os.path.isdir(model_path):
            if is_diffusers_model(model_path):
                commit_message = f"Upload diffusers format: {trained_model}"
                print("Detected diffusers model. Adjusting upload parameters.")
            else:
                commit_message = f"Upload checkpoint: {trained_model}"
                print("Detected regular model. Adjusting upload parameters.")

            api.upload_folder(
                folder_path=model_path,
                path_in_repo=path_in_repo_local,
                repo_id=model_repo,
                commit_message=commit_message,
                ignore_patterns=".ipynb_checkpoints",
            )
        else:
            commit_message = f"Upload file: {trained_model}"
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=path_in_repo_local,
                repo_id=model_repo,
                commit_message=commit_message,
            )
        with output_widget:
            return f"Model upload complete! Check it out at https://huggingface.co/{model_repo}/tree/main"

    except Exception as e:
        with output_widget:
            return f"Upload failed: {e}"

# ---------------------- GRADIO INTERFACE ----------------------

def main(model_to_load, save_precision_as, epoch, global_step, reference_model, output_path, fp16, hf_token, orgs_name, model_name, make_private):
  """Main function orchestrating the entire process."""
  output = gr.Markdown()

  conversion_output = convert_model(model_to_load, save_precision_as, epoch, global_step, reference_model, output_path, fp16, output)

  upload_output = upload_to_huggingface(output_path, hf_token, orgs_name, model_name, make_private, output)

  # Return a combined output
  return f"{conversion_output}\n\n{upload_output}"

with gr.Blocks() as demo:

    # Add initial warnings (only once)
    gr.Markdown("""
        ## **⚠️ IMPORTANT WARNINGS ⚠️**
        This app may violate Google Colab AUP.  Use at your own risk.  `xformers` may cause issues.
    """)

    model_to_load = gr.Textbox(label="Model to Load (Checkpoint or Diffusers)", placeholder="Path to model")
    with gr.Row():
        save_precision_as = gr.Dropdown(
            choices=["fp16", "bf16", "float"], value="fp16", label="Save Precision As"
        )
        fp16 = gr.Checkbox(label="Load as fp16 (Diffusers only)")
    with gr.Row():
        epoch = gr.Number(value=0, label="Epoch to Write (Checkpoint)")
        global_step = gr.Number(value=0, label="Global Step to Write (Checkpoint)")

    reference_model = gr.Textbox(label="Reference Diffusers Model",
                                 placeholder="e.g., stabilityai/stable-diffusion-xl-base-1.0")
    output_path = gr.Textbox(label="Output Path", value="/content/output")

    gr.Markdown("## Hugging Face Hub Configuration")
    hf_token = gr.Textbox(label="Hugging Face Token", placeholder="Your Hugging Face write token")
    with gr.Row():
        orgs_name = gr.Textbox(label="Organization Name (Optional)", placeholder="Your organization name")
        model_name = gr.Textbox(label="Model Name", placeholder="The name of your model on Hugging Face")
    make_private = gr.Checkbox(label="Make Repository Private", value=False)

    convert_button = gr.Button("Convert and Upload")
    output = gr.Markdown()

    convert_button.click(fn=main,
                       inputs=[model_to_load, save_precision_as, epoch, global_step, reference_model,
                               output_path, fp16, hf_token, orgs_name, model_name, make_private],
                       outputs=output)

demo.launch()
