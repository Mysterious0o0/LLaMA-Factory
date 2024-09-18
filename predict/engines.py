from typing import List, Tuple, Dict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest


class Qwen72ChatEngine:
    def __init__(self, model_path, lora_path, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.5,
            seed=42,
            enable_lora=True,
            max_model_len=1500,
            **kwargs,
        )
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=100)
        self.lora_request = LoRARequest("cosplay", 1, lora_path=lora_path)

    def _build_message(self, system: str, histories: List[Tuple[str, str]], query: str) -> List[Dict]:
        messages = []
        system_format = {"role": "system", "content": f"{system}"}
        messages.append(system_format)
        for user_query, assistant in histories:
            messages.append({"role": "user", "content": f"{user_query}"})
            messages.append({"role": "assistant", "content": f"{assistant}"})
        messages.append({"role": "user", "content": f"{query}"})
        return messages

    def chat(self, system: str, histories: List[Tuple[str, str]], query: str):
        messages = self._build_message(system, histories, query)

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = self.llm.generate([text], self.sampling_params, lora_request=self.lora_request)
        for output in outputs:
            # prompt = output.prompt
            response = output.outputs[0].text
            print(response)
        return response
