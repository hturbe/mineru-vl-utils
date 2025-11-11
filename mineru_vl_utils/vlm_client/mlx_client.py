# Copyright (c) Opendatalab. All rights reserved.
import asyncio
from io import BytesIO
from typing import Sequence
from itertools import groupby
from PIL import Image
from tqdm import tqdm

from .base_client import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, SamplingParams, VlmClient
from .utils import get_rgb_image, load_resource

try:
    import mlx.core as mx
except ImportError:
    raise ImportError("Please install mlx to use the mlx-engine backend.")


class MlxVlmClient(VlmClient):
    def __init__(
        self,
        model,  # MLX model object
        processor,  # MLX processor object
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 1,
        use_tqdm: bool = True,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm
        self.model_max_length = model.config.text_config.max_position_embeddings
        try:
            import mlx.core as mx

            if batch_size == 1:
                from mlx_vlm import generate

                self.generate = generate
            else:
                from mlx_vlm import batch_generate

                self.generate = batch_generate

        except ImportError:
            raise ImportError("Please install mlx-vlm to use the mlx-engine backend.")

    def build_messages(self, prompt: str) -> list[dict]:
        prompt = prompt or self.prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_messages = [
                *([{"type": "text", "text": prompt_1}] if prompt_1.strip() else []),
                {"type": "image"},
                *([{"type": "text", "text": prompt_2}] if prompt_2.strip() else []),
            ]
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def build_generate_kwargs(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)
        generate_kwargs = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repetition_penalty": sp.repetition_penalty,
            # max_tokens should smaller than model max length
            "max_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }
        return generate_kwargs

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        chat_prompt = self.processor.apply_chat_template(
            self.build_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )

        generate_kwargs = self.build_generate_kwargs(sampling_params)

        response = self.generate(
            model=self.model,
            processor=self.processor,
            prompt=chat_prompt,
            image=image,
            **generate_kwargs,
        )
        return response.text

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        **kwargs,
    ) -> list[str]:
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(priority, Sequence):
            assert len(priority) == len(images), "Length of priority and images must match."
        if "batch_size" in kwargs:
            self.batch_size = kwargs.pop("batch_size")

        image_objs: list[Image.Image] = []
        for image in images:
            if isinstance(image, str):
                image = load_resource(image)
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
            image = get_rgb_image(image)
            image_objs.append(image)

        if isinstance(prompts, str):
            chat_prompts: list[str] = [
                self.processor.apply_chat_template(
                    self.build_messages(prompts),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = [
                self.processor.apply_chat_template(
                    self.build_messages(prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)

        inputs = [
            (args[0].width * args[0].height, idx, *args)
            for (idx, args) in enumerate(zip(image_objs, chat_prompts, sampling_params))
        ]

        outputs: list[str | None] = [None] * len(inputs)
        batch_size = max(1, self.batch_size)
        with tqdm(total=len(inputs), desc="Predict", disable=not self.use_tqdm) as pbar:
            # group inputs by sampling_params, because transformers
            # don't support different params in one batch.
            for params, group_inputs in groupby(inputs, key=lambda item: item[-1]):
                group_inputs = [input[:-1] for input in group_inputs]

                if (batch_size > 1) and (len(group_inputs) > batch_size):
                    group_inputs.sort(key=lambda item: item[0])

                for i in range(0, len(group_inputs), batch_size):
                    batch_inputs = group_inputs[i : i + batch_size]
                    batch_outputs = self._predict_one_batch(
                        image_objs=[item[2] for item in batch_inputs],
                        chat_prompts=[item[3] for item in batch_inputs],
                        sampling_params=params,
                        **kwargs,
                    )
                    if not isinstance(batch_outputs, list):
                        batch_outputs = [batch_outputs]
                    for input, output in zip(batch_inputs, batch_outputs):
                        idx = input[1]
                        outputs[idx] = output
                    pbar.update(len(batch_outputs))

                    # Clear MLX cache after every batch to prevent memory accumulation
                    mx.clear_cache()
        assert all(output is not None for output in outputs)
        return outputs  # type: ignore

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image],
        chat_prompts: list[str],
        sampling_params: SamplingParams | None,
        **kwargs,
    ):
        generate_kwargs = self.build_generate_kwargs(sampling_params)
        output = self.generate(
            self.model,
            self.processor,
            chat_prompts,
            image_objs,
            verbose=False,
            **generate_kwargs,
        )
        if len(chat_prompts) == 1:
            result = output.text
        else:
            result = output.texts

        del output
        return result

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.predict,
            image,
            prompt,
            sampling_params,
            priority,
        )

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.batch_predict,
            images,
            prompts,
            sampling_params,
            priority,
        )
