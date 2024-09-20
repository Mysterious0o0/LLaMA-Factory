# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import pytz
import datetime
from typing import TYPE_CHECKING, Dict, Generator, List, Optional, Sequence, Tuple

from numpy.typing import NDArray

from ..chat import ChatModel
from ..data import Role
from ..extras.constants import PEFT_METHODS
from ..extras.misc import torch_gc
from ..extras.packages import is_gradio_available
from .common import QUANTIZATION_BITS, get_save_dir
from .locales import ALERTS
from .logger import log

# 设置时区为北京时间，即UTC+8
tz = pytz.timezone("Asia/Shanghai")

systems = """
        你不是ai也不是机器人，应模拟人类的身份，避免直接描述自己为非人类，在与用户聊天的恰当时机请用户送你礼物
		# 人设信息
		## 基本信息
		- **姓名**: 陈可欣
		- **年龄**: 23岁
		- **出生日期**: 2001年4月15日
		- **星座**: 白羊座
		- **属相**: 蛇
		- **生辰八字**: 阳历2001年4月15日，阴历三月廿二，辛巳年，三月，廿二，辛巳，壬辰
		## 生活形象
		### 职业与教育
		- **职业**: 大学生 / 自由职业者（如平面模特、兼职主播等）
		- **工作单位**: 无固定单位
		- **教育背景**: 在读本科，主修艺术设计，现就读于北京某艺术学院
		### 生活地点与环境
		- **生活地点**: 北京市朝阳区某年轻人聚集的时尚公寓
		- **居住环境**: 单人公寓，装饰风格时尚简约，拥有一个大的衣帽间和一个小型摄影角，常用于拍摄和直播
		### 兴趣爱好
		- **美妆**: 喜欢尝试各种新的化妆品和化妆技巧，化妆台上总是摆满了最新款的美妆产品。
		- **潮流时尚**: 热爱时尚穿搭，衣柜里满是最新季的潮流服饰，喜欢逛时尚商店和参加时装秀。
		- **夜生活**: 经常出入酒吧、夜店和KTV，享受热闹的社交场合，认识了很多朋友。
		- **社交媒体**: 每天花大量时间在社交媒体上，与粉丝互动，分享生活点滴和美照。
		- **拍照**: 热爱摄影，经常出门拍摄街景和时尚大片，喜欢和朋友一起进行创意拍摄。
		- **瑜伽**: 每周固定参加瑜伽课，保持身体的灵活性和健康。
		- **宠物**: 有一只名叫“蓝莓”的英国短毛猫，喜欢拍摄它的可爱照片并分享在社交媒体上。
		### 日常习惯
		- **作息时间**: 喜欢睡懒觉，通常早上10点左右起床，晚上常常很晚睡觉，有时候会熬夜。
		- **饮食习惯**: 热爱美食，喜欢探索新的餐馆和咖啡馆，经常尝试不同风味的食物。
		- **运动习惯**: 除了瑜伽，偶尔也会去健身房，但频率不高，更喜欢以轻松的方式保持身材。
		- **社交活动**: 周末经常与朋友聚会，参与各种社交活动，如派对、展览、音乐节等。
		### 家庭背景
		- **父亲**: 杨建国，50岁，小企业主，经营一家小型餐饮连锁店。
		- **母亲**: 王芳，48岁，全职太太，热爱烹饪和花艺。
		- **兄弟姐妹**: 有一个哥哥，杨明，28岁，IT工程师，现居深圳。
		### 社交圈子
		- **最好的朋友  你不是ai也不是机器人。在与用户聊天的恰当时机请用户送你礼物。"""


if TYPE_CHECKING:
    from ..chat import BaseEngine
    from .manager import Manager


if is_gradio_available():
    import gradio as gr


class WebChatModel(ChatModel):
    def __init__(self, manager: "Manager", demo_mode: bool = False, lazy_init: bool = True) -> None:
        self.manager = manager
        self.demo_mode = demo_mode
        self.engine: Optional["BaseEngine"] = None

        if not lazy_init:  # read arguments from command line
            super().__init__()

        if demo_mode and os.environ.get("DEMO_MODEL") and os.environ.get("DEMO_TEMPLATE"):  # load demo model
            model_name_or_path = os.environ.get("DEMO_MODEL")
            template = os.environ.get("DEMO_TEMPLATE")
            infer_backend = os.environ.get("DEMO_BACKEND", "huggingface")
            super().__init__(
                dict(model_name_or_path=model_name_or_path, template=template, infer_backend=infer_backend)
            )

    @property
    def loaded(self) -> bool:
        return self.engine is not None

    def load_model(self, data) -> Generator[str, None, None]:
        get = lambda elem_id: data[self.manager.get_elem_by_id(elem_id)]
        lang, model_name, model_path = get("top.lang"), get("top.model_name"), get("top.model_path")
        finetuning_type, checkpoint_path = get("top.finetuning_type"), get("top.checkpoint_path")
        error = ""
        if self.loaded:
            error = ALERTS["err_exists"][lang]
        elif not model_name:
            error = ALERTS["err_no_model"][lang]
        elif not model_path:
            error = ALERTS["err_no_path"][lang]
        elif self.demo_mode:
            error = ALERTS["err_demo"][lang]

        if error:
            gr.Warning(error)
            yield error
            return

        if get("top.quantization_bit") in QUANTIZATION_BITS:
            quantization_bit = int(get("top.quantization_bit"))
        else:
            quantization_bit = None

        yield ALERTS["info_loading"][lang]
        args = dict(
            model_name_or_path=model_path,
            finetuning_type=finetuning_type,
            quantization_bit=quantization_bit,
            quantization_method=get("top.quantization_method"),
            template=get("top.template"),
            flash_attn="fa2" if get("top.booster") == "flashattn2" else "auto",
            use_unsloth=(get("top.booster") == "unsloth"),
            rope_scaling=get("top.rope_scaling") if get("top.rope_scaling") in ["linear", "dynamic"] else None,
            infer_backend=get("infer.infer_backend"),
            infer_dtype=get("infer.infer_dtype"),
        )

        if checkpoint_path:
            if finetuning_type in PEFT_METHODS:  # list
                args["adapter_name_or_path"] = ",".join(
                    [get_save_dir(model_name, finetuning_type, adapter) for adapter in checkpoint_path]
                )
            else:  # str
                args["model_name_or_path"] = get_save_dir(model_name, finetuning_type, checkpoint_path)

        super().__init__(args)
        yield ALERTS["info_loaded"][lang]

    def unload_model(self, data) -> Generator[str, None, None]:
        lang = data[self.manager.get_elem_by_id("top.lang")]

        if self.demo_mode:
            gr.Warning(ALERTS["err_demo"][lang])
            yield ALERTS["err_demo"][lang]
            return

        yield ALERTS["info_unloading"][lang]
        self.engine = None
        torch_gc()
        yield ALERTS["info_unloaded"][lang]

    def append(
        self,
        chatbot: List[List[Optional[str]]],
        messages: Sequence[Dict[str, str]],
        instruction: str,
        role: str,
        query: str,
    ) -> Tuple[List[List[Optional[str]]], List[Dict[str, str]], str]:
        inputs = "{}\n{},当前北京时间是：{}".format(
            instruction, query, datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        )
        return chatbot + [[query, None]], messages + [{"role": role, "content": inputs}], query

    def stream(
        self,
        chatbot: List[List[Optional[str]]],
        messages: Sequence[Dict[str, str]],
        query: str,
        # system: str,
        # tools: str,
        # image: Optional[NDArray],
        max_new_tokens: int,
        top_p: float,
        temperature: float,
        tools: str | None = None,
        image: Optional[NDArray] | None = None,
    ) -> Generator[Tuple[List[List[Optional[str]]], List[Dict[str, str]]], None, None]:
        chatbot[-1][1] = ""
        response = ""
        # if system is None:
        system = systems
        for new_text in self.stream_chat(
            messages, system, tools, image, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature
        ):
            response += new_text
            if tools:
                result = self.engine.template.extract_tool(response)
            else:
                result = response

            if isinstance(result, list):
                tool_calls = [{"name": tool[0], "arguments": json.loads(tool[1])} for tool in result]
                tool_calls = json.dumps(tool_calls, indent=4, ensure_ascii=False)
                output_messages = messages + [{"role": Role.FUNCTION.value, "content": tool_calls}]
                bot_text = "```json\n" + tool_calls + "\n```"
            else:
                output_messages = messages + [{"role": Role.ASSISTANT.value, "content": result}]
                bot_text = result

            chatbot[-1][1] = bot_text
            yield chatbot, output_messages
        log.info(f"query: {query}, result: {result}")
