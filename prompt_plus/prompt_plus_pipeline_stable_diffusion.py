import os
import json
from typing import Optional, List, Union, Callable, Dict, Any, Tuple
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging
from huggingface_hub import hf_hub_download
from prompt_plus.prompt_plus_unet_2d_condition import PPlusUNet2DConditionModel


logger = logging.get_logger(__name__)


class TextualInversionStableDiffusionPipeline(StableDiffusionPipeline):
    @classmethod
    def from_learned_embed(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            learned_embed_name_or_path: Union[str, os.PathLike],
            **kwargs
    ):
        if os.path.exists(learned_embed_name_or_path):
            embeds_path = os.path.join(learned_embed_name_or_path, "learned_embeds.bin") if os.path.isdir(learned_embed_name_or_path) else learned_embed_name_or_path
            # token_path = os.path.join(model_dir, "token_identifier.txt")
        else:
            # download
            embeds_path = hf_hub_download(repo_id=learned_embed_name_or_path, filename="learned_embeds.bin")
            # token_path = hf_hub_download(repo_id=learned_embed_name_or_path, filename="token_identifier.txt")

        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", **kwargs
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", **kwargs)
        loaded_learned_embeds = torch.load(embeds_path, map_location="cpu")
        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        embeds.to(dtype)

        # add the token in tokenizer
        # token = token if token is not None else trained_token
        num_added_tokens = tokenizer.add_tokens(trained_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {trained_token}. Please pass a different `token` that is not already in the tokenizer.")

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(trained_token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
        print(f"placeholder_token: {trained_token}")
        return super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            **kwargs
        )


def _load_embed_from_name_or_path(learned_embed_name_or_path):
    if os.path.exists(learned_embed_name_or_path):
        embeds_path = os.path.join(learned_embed_name_or_path, "learned_embeds.bin") if os.path.isdir(
            learned_embed_name_or_path) else learned_embed_name_or_path
        # config_path = os.path.join(model_dir, "config.json")
    else:
        # download
        embeds_path = hf_hub_download(repo_id=learned_embed_name_or_path, filename="learned_embeds.bin")
        # config_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.json")
    # with open(config_path, "r", encoding="utf-8") as f:
    #     config = json.load(f)
    # load
    loaded_learned_embeds = torch.load(embeds_path, map_location="cpu")
    return loaded_learned_embeds


def load_embed_from_name_or_path(learned_embed_name_or_path, style_mixing_k_K=None):
    if isinstance(learned_embed_name_or_path, str):
        assert style_mixing_k_K is None, "You inputted only one learned embed but `style_mixing_k_K` was specified!"
        return _load_embed_from_name_or_path(learned_embed_name_or_path)
    else:
        assert len(learned_embed_name_or_path) == 2, "Only 2 embeds are supported for now but it's especially possible."
        k, K = style_mixing_k_K
        embeds = []
        for p in learned_embed_name_or_path:
            embeds.append(_load_embed_from_name_or_path(p))
        # use first embeds tokens to align
        tokens = list(embeds[0].keys())
        n = len(tokens)
        assert k < n, f"k must be lower than n={n}"
        assert K < n, f"K must be lower than n={n}"
        loaded_learned_embeds = dict()
        for i in range(n):
            if i <= k or K > i:
                embed_idx = 0
            else:
                embed_idx = 1
            embed = list(embeds[embed_idx].values())[i]
            loaded_learned_embeds[tokens[i]] = embed
        return loaded_learned_embeds


class PPlusStableDiffusionPipeline(StableDiffusionPipeline):
    @classmethod
    def from_learned_embed(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            learned_embed_name_or_path: Optional[Union[str, os.PathLike, List[str]]] = None,
            style_mixing_k_K: Optional[Tuple[int]] = None,
            loaded_learned_embeds: Optional[Dict[str, torch.Tensor]] = None,
            **kwargs,
    ):
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", **kwargs
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", **kwargs)
        if loaded_learned_embeds is None:
            loaded_learned_embeds = load_embed_from_name_or_path(learned_embed_name_or_path, style_mixing_k_K)
        new_tokens = list(loaded_learned_embeds.keys())
        # easy validation for textual inversion
        assert len(new_tokens) > 1, "You might want to load textual inversion pipeline!"
        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer)+len(new_tokens))

        for token in new_tokens:
            embeds = loaded_learned_embeds[token]
            embeds.to(dtype)
            # add the token in tokenizer
            # token = token if token is not None else trained_token
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
            # get the id for the token and assign the embeds
            token_id = tokenizer.convert_tokens_to_ids(token)
            text_encoder.get_input_embeddings().weight.data[token_id] = loaded_learned_embeds[token]
        # store placeholder_token to text_encoder config
        text_encoder.config.placeholder_token = "-".join(new_tokens[0].split("-")[:-1])
        text_encoder.config.placeholder_tokens = new_tokens
        print(f"placeholder_token: {text_encoder.config.placeholder_token}")
        unet = PPlusUNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", **kwargs)
        return super().from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            unet=unet,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            **kwargs
        )

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        assert isinstance(prompt, str), "Currently, only string `prompt` is supported!"
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            encoder_hidden_states_list = []
            for token in self.text_encoder.config.placeholder_tokens:
                one_prompt = prompt.replace(self.text_encoder.config.placeholder_token, token)
                text_inputs = self.tokenizer(
                    one_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(one_prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                        text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if hasattr(self.text_encoder.config,
                           "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device),
                    attention_mask=attention_mask,
                )
                prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

                # get unconditional embeddings for classifier free guidance
                if do_classifier_free_guidance:
                    uncond_tokens: List[str]
                    if negative_prompt is None:
                        uncond_tokens = [""] * batch_size
                    elif type(prompt) is not type(negative_prompt):
                        raise TypeError(
                            f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                            f" {type(prompt)}."
                        )
                    elif isinstance(negative_prompt, str):
                        uncond_tokens = [negative_prompt]
                    elif batch_size != len(negative_prompt):
                        raise ValueError(
                            f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                            f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                            " the batch size of `prompt`."
                        )
                    else:
                        uncond_tokens = negative_prompt

                    max_length = prompt_embeds.shape[1]
                    uncond_input = self.tokenizer(
                        uncond_tokens,
                        padding="max_length",
                        max_length=max_length,
                        truncation=True,
                        return_tensors="pt",
                    )

                    if hasattr(self.text_encoder.config,
                               "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                        attention_mask = uncond_input.attention_mask.to(device)
                    else:
                        attention_mask = None

                    negative_prompt_embeds = self.text_encoder(
                        uncond_input.input_ids.to(device),
                        attention_mask=attention_mask,
                    )
                    negative_prompt_embeds = negative_prompt_embeds[0]

                    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                    seq_len = negative_prompt_embeds.shape[1]

                    negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len,
                                                                         -1)

                    # For classifier free guidance, we need to do two forward passes.
                    # Here we concatenate the unconditional and text embeddings into a single batch
                    # to avoid doing two forward passes
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                encoder_hidden_states_list.append(prompt_embeds)
        else:
            # trust you!
            encoder_hidden_states_list = prompt_embeds
        return encoder_hidden_states_list

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        encoder_hidden_states_list = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            encoder_hidden_states_list[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states_list=encoder_hidden_states_list,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, encoder_hidden_states_list[0].dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, encoder_hidden_states_list[0].dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

