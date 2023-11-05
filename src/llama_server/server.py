import asyncio
import base64
import json
import math
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import aiohttp
import click
from loguru import logger
from PIL import Image

log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green>: <b>{message}</b>"
logger.remove()
logger.add(sys.stderr, level="INFO", format=log_format, colorize=True, backtrace=True, diagnose=True)


def scale_image(img: Image.Image | Path | str, base_width=300):
    if isinstance(img, Path | str):
        img = Image.open(img)
    wpercent = base_width / float(img.size[0])
    hsize = int(float(img.size[1]) * float(wpercent))
    img = img.resize((base_width, hsize), Image.Resampling.LANCZOS)
    return img


def encode_image(img: Image.Image):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode()


class LlamaServer:
    def __init__(
        self,
        config: dict | Path | str,
        context_size: int = 4096,
        batch_size: int = 512,
        slots: int = 5,
    ):
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = json.load(Path(config).as_posix())
        self.model = Path(config["model"])
        self.server_exe = Path(config["server_exe"])
        self.alias = config.get("alias", "Llama")
        self.multimodal_projector = config.get("multimodal_projector", None)
        if self.multimodal_projector is not None:
            self.multimodal_projector = Path(self.multimodal_projector)
            self.image_width = config.get("image_width", 300)
            self.image_tag = config.get("image_tag", "mmtag")
            if slots > 1:
                logger.info("Multimodal inference currently only works with one slot. Slot number is changed to 1.")
                slots = 1

        self.system_prompt = config.get("system_prompt", None)
        if self.system_prompt is not None:
            self.system_prompt = Path(self.system_prompt)

        self.prefix = config.get("prefix", "")
        self.suffix = config.get("suffix", "")

        self.context_size = context_size
        self.batch_size = batch_size
        self.slots = slots
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 8080)
        self.server_url = f"http://{self.host}:{self.port}"
        self.server_process = None

    async def start_server(self, wait_online: int = 300):
        if getattr(self, "server_process", None) is not None:
            if self.server_process.poll() is None:
                logger.info("Llama server is already online.")
                return self.server_process

        self.server_command = [
            self.server_exe.resolve(),
            "--model",
            self.model.as_posix(),
            "--alias",
            self.alias,
            "--ctx-size",
            f"{self.context_size}",
            "--batch-size",
            f"{self.batch_size}",
            "--main-gpu",
            "0",
            "--tensor-split",
            "5,5",
            "--n-gpu-layers",
            f"{1024}",  # set to a large value
            "--parallel",
            f"{self.slots}",
            "--host",
            f"{self.host}",
            "--port",
            f"{self.port}",
            "--cont-batching",
        ]

        if self.system_prompt is not None:
            self.server_command += [
                "--system-prompt-file",
                self.system_prompt.as_posix(),
            ]

        if self.multimodal_projector is not None:
            self.server_command += [
                "--mmproj",
                self.multimodal_projector.as_posix(),
            ]

        self.server_process = subprocess.Popen(
            self.server_command, shell=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        online = False
        async with aiohttp.ClientSession() as session:
            for i in range(wait_online):
                await asyncio.sleep(1)  # sleep 1s
                try:
                    response = await session.get(f"{self.server_url}/props")  # check if server is online
                    if response.status == 200:
                        online = True
                        break
                except aiohttp.ClientConnectorError:
                    if i + 1 % 5 == 0:
                        logger.info(f"Wait for server online ({i+1}/{wait_online} seconds).")

        if not online:
            raise RuntimeError(f"Failed to start llams server with {self.server_command}.")
        else:
            logger.info("Server online.")
        return self.server_process

    def stop_server(self):
        if getattr(self, "server_process", None) is None:
            logger.info("Llama server is already offline.")
            return 0
        elif self.server_process.poll() is not None:
            self.server_process = None
            logger.info("Llama server has crashed.")
            return 0
        else:
            self.server_process.terminate()
            # Wait for process to terminate
            returncode = self.server_process.wait()
            self.server_process = None
            return returncode

    async def tokenize(self, content: str):
        api_url = f"{self.server_url}/tokenize"
        headers = {"Content-Type": "application/json"}
        data = {"content": content}
        async with aiohttp.ClientSession() as session:
            response = await session.post(api_url, json=data, headers=headers)
            output = await response.json()
        return output

    async def detokenize(self, tokens: str):
        api_url = f"{self.server_url}/detokenize"
        headers = {"Content-Type": "application/json"}
        data = {"tokens": tokens}
        async with aiohttp.ClientSession() as session:
            response = await session.post(api_url, json=data, headers=headers)
            output = await response.json()
        return output

    async def _query_server(
        self,
        prompt: str,
        image: Image.Image | None = None,
        *,
        slot_id: int = -1,
        stop: list = [],
        temperature: float = 0.2,
        seed: int = -1,
        n_predict: int = 512,
        cache_prompt: bool = False,
        system_prompt: str = "",
        grammar: str = "",
    ) -> dict:
        api_url = f"{self.server_url}/completion"
        headers = {"Content-Type": "application/json"}

        data = {
            "prompt": self.prefix + prompt + self.suffix,
            "temperature": temperature,
            "seed": seed,
            "n_predict": n_predict,
            "stop": stop,
            "slot_id": slot_id,
            "cache_prompt": cache_prompt,
        }

        if grammar != "":
            data["grammar"] = grammar

        if image is not None:
            encoded_string = encode_image(image)
            img_id = 100
            data["image_data"] = [{"data": encoded_string, "id": img_id}]

            if self.image_tag == "mmtag":
                image_encode = f"<Image>[img-{img_id}]</Image>"
            else:
                image_encode = f"[img-{img_id}]"
            data["prompt"] = self.prefix + image_encode + "\n" + prompt + self.suffix

        if system_prompt != "":
            data = {"system_prompt": system_prompt}

        async with aiohttp.ClientSession() as session:
            try:
                response = await session.post(api_url, data=json.dumps(data), headers=headers)
                response.raise_for_status()
                output = await response.json()
                return output

            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection error: {e}")
        return {}

    async def query(self, prompt, image: Image.Image | Path | str = None, **kwargs):
        if image is not None:
            image = scale_image(image, base_width=self.image_width)
        result = await self._query_server(prompt, image, **kwargs)
        return result, image

    async def batch_query(self, prompts, images: list[Image.Image | Path | str] | None = None, **kwargs) -> list[dict]:
        """Query the LLM in batches

        Parameters
        ----------
        prompts : List
            A list of prompt strings.
        **kwargs
            Parameters passed to _query_server function.

        Returns
        -------
        responses : List[Dict]
            A list of LLM responses.
        """
        if images is not None:
            images = [scale_image(image, base_width=self.image_width) for image in images]
        else:
            images = [None] * len(prompts)

        n_batch = math.ceil(len(prompts) / self.slots)

        results = []
        for batch_num in range(n_batch):
            queries = []
            batch_prompts = prompts[batch_num * self.slots : (batch_num + 1) * self.slots]
            batch_images = images[batch_num * self.slots : (batch_num + 1) * self.slots]

            for i, (image, prompt) in enumerate(zip(batch_images, batch_prompts)):
                queries.append(self._query_server(prompt, image, slot_id=i % self.slots, **kwargs))
            results += await asyncio.gather(*queries)

        return results, images

    async def update_system_prompt(self, system_prompt: dict[str, str]):
        result = await self._query_server("", system_prompt=system_prompt)
        return result != {}  # result = {} on request failure.


@click.command()
@click.option("--slots", default=1, type=click.INT)
def test(slots: int = 1):
    logger.add(
        f"cont-batching-slot-{slots}.log",
        level="DEBUG",
        format=log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )
    config = {
        "alias": "LLaMa-2-chat",
        "model": "./llama-2-7b-chat.Q5_K_M.gguf",
        "server_exe": "./llama_server",
        "system_prompt": "./llama2_prompt.json",
        "prefix": "[INST]",
        "suffix": "[/INST]",
    }

    server = LlamaServer(config, context_size=2048, slots=slots)
    logger.info("Starting Llama server.")
    p = asyncio.run(server.start_server())
    logger.info(f"Server pid: {p.pid}")

    prompts = [
        "How is the weather?",
        "Where is the captical of China?",
        "Who is the president of the USA?",
        "How old are you?",
        "What is the value of 1+1?",
        r"How to calculate \pi?",
        "How to teach math in university?",
        "How to build a website?",
        r"sin(\pi)+cos(\pi)=?",
        r"How to plot a sin wave in range [-\pi,\pi] with Julia lang?",
    ]

    logger.info("Start evaluation.")
    start = time.time()
    responses, _ = asyncio.run(server.batch_query(prompts))  # no image input
    end = time.time()
    logger.info(f"Time elapsed: {end - start:.4f} seconds")

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        answer = response["content"].strip()
        logger.debug(f"Conversation {i+1}.\nQ: {prompt}\nA: {answer}\n")

    server.server_process.terminate()
    server.server_process.wait()
    logger.info("Server is closed.")


if __name__ == "__main__":
    test()
