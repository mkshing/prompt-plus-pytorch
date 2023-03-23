import sys
import os
from functools import lru_cache
import subprocess
import torch
import gradio as gr
from diffusers.utils import is_xformers_available


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


def gitclone(url, target_dir=None, branch_arg=None):
    run_args = ["git", "clone"]
    if branch_arg:
        run_args.extend(["-b", branch_arg])
    run_args.append(url)
    if target_dir:
        run_args.append(target_dir)
    res = subprocess.run(run_args, stdout=subprocess.PIPE).stdout.decode("utf-8")
    print(res)


def pipi(modulestr):
    res = subprocess.run(
        ["pip", "install", modulestr], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    print(res)


try:
    proj_dir = os.path.dirname(__file__)
    sys.path.append(proj_dir)
    from prompt_plus import PPlusStableDiffusionPipeline
except ImportError:
    GITHUB_SECRET = os.environ.get("GITHUB_SECRET")
    gitclone("https://github.com/mkshing/prompt-plus-pytorch" if GITHUB_SECRET is None else f"https://{GITHUB_SECRET}@github.com/mkshing/prompt-plus-pytorch")
    from prompt_plus import PPlusStableDiffusionPipeline


@lru_cache(maxsize=3)
def load_pipe(pretrained_model_name_or_path, learned_embed_name_or_path):
    pipe = PPlusStableDiffusionPipeline.from_learned_embed(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        learned_embed_name_or_path=learned_embed_name_or_path,
        revision="fp16", torch_dtype=torch.float16
    )
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    return pipe


def txt2img_func(pretrained_model_name_or_path, learned_embed_name_or_path, prompt, n_samples=4, scale=7.5, steps=25, width=512, height=512, seed="random"):
    n_samples = int(n_samples)
    scale = float(scale)
    steps = int(steps)
    width = int(width)
    height = int(height)
    generator = torch.Generator(device=device)
    if seed == "random":
        seed = generator.seed()
    else:
        seed = int(seed)
    generator = generator.manual_seed(int(seed))
    pipe = load_pipe(pretrained_model_name_or_path, learned_embed_name_or_path).to(device)
    images = pipe(
        prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        generator=generator,
        num_images_per_prompt=n_samples,
        height=height,
        width=width
    ).images
    return images


with gr.Blocks() as demo:
    gr.Markdown("# P+: Extended Textual Conditioning in Text-to-Image Generation")
    pretrained_model_name_or_path = gr.Textbox(label="pre-trained model name or path", value="runwayml/stable-diffusion-v1-5")
    learned_embed_name_or_path = gr.Textbox(label="learned embedding name or path")
    with gr.Row():
        with gr.Column():
            # input
            prompt = gr.Textbox(label="Prompt")
            n_samples = gr.Number(value=3, label="n_samples")
            cfg_scale = gr.Slider(minimum=0.0, maximum=20, value=7.5, label="cfg_scale", step=0.5)
            steps = gr.Number(value=30, label="steps")
            width = gr.Slider(minimum=128, maximum=1024, value=512, label="width", step=64)
            height = gr.Slider(minimum=128, maximum=1024, value=512, label="height", step=64)
            seed = gr.Textbox(value='random',
                              placeholder="If you fix seed, you get same outputs all the time. You can set as integer like 42.",
                              label="seed")

            # button
            button = gr.Button(value="Generate!")
        with gr.Column():
            # output
            out_images = gr.Gallery(label="Output")
    button.click(
        txt2img_func,
        inputs=[pretrained_model_name_or_path, learned_embed_name_or_path, prompt, n_samples, cfg_scale, steps, width, height, seed],
        outputs=[out_images],
        api_name="txt2img"
    )

demo.launch()
