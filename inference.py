import argparse
from PIL import Image
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.utils import is_xformers_available
from prompt_plus import TextualInversionStableDiffusionPipeline, PPlusStableDiffusionPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, help="model name or path", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--learned_embed_name_or_path", type=str, help="model path for learned embedding")
    parser.add_argument("--is_textual_inversion", action="store_true", help="Load textual inversion embeds")
    parser.add_argument("--original_pipe", action="store_true", help="load standard pipeline")
    parser.add_argument("--device", type=str, help="Device on which Stable Diffusion will be run", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--float16", action="store_true", help="load float16")
    # diffusers config
    parser.add_argument("--prompt", type=str, nargs="?", default="a photo of *s", help="the prompt to render")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="number of ddim sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="unconditional guidance scale")
    parser.add_argument("--num_images_per_prompt", type=int, default=3, help="number of images per prompt")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space",)
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space",)
    parser.add_argument("--seed", type=int, default=None, help="the seed (for reproducible sampling)")
    opt = parser.parse_args()
    return opt


def main():
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {args.device}")

    # load model
    if args.is_textual_inversion or not args.original_pipe:
        if args.is_textual_inversion:
            Pipeline = TextualInversionStableDiffusionPipeline
        else:
            Pipeline = PPlusStableDiffusionPipeline
        pipe = Pipeline.from_learned_embed(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            learned_embed_name_or_path=args.learned_embed_name_or_path,
            torch_dtype=torch.float16 if args.float16 else None,
        ).to(args.device)
    else:
        print("loading the original pipeline")
        pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16 if args.float16 else None).to(args.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    if is_xformers_available():
        pipe.enable_xformers_memory_efficient_attention()
    print("loaded pipeline")
    # run!
    generator = None
    if args.seed:
        print(f"Using seed: {args.seed}")
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
    images = pipe(
        args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        num_images_per_prompt=args.num_images_per_prompt,
        height=args.height,
        width=args.width
    ).images
    grid_image = image_grid(images, 1, args.num_images_per_prompt)
    grid_image.save("grid.png")
    print("DONE!")


if __name__ == '__main__':
    main()

