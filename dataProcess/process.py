import json
import pandas as pd
from glob import glob
from random import shuffle
from time import perf_counter
from typing import List, Tuple
from dataclasses import dataclass, field

from logger import log


@dataclass
class TrainData:
    instruction: str = field(default=None)
    query: str = field(default=None)
    output: str = field(default=None)
    system: str = field(
        default="你不是ai也不是机器人，应模拟人类的身份，避免直接描述自己为非人类，在与用户聊天的恰当时机请用户送你礼物"
    )
    history: list = field(default=None)


def get_instruction_cls(path: str) -> int:
    if "普通" in path:
        return 0
    else:
        return 1


def construct_data(
    query_parts: List, answer_parts: List, history: List[Tuple[str, str]], num: int, instruction_cls: int
) -> Tuple[str, str, TrainData]:
    query = " ".join(query_parts)
    answer = " ".join(answer_parts)
    match instruction_cls:
        case 0:
            instruction = "你是一个女大学生，你将进行网络私聊的对话，请根据你的基本信息和生活形象，以轻松、友好、含蓄、好奇、简洁、适度赞美、保持一定神秘感并保持话题连贯性的口吻回复用户，在适当的时候可以索要礼物"
        case 1:
            instruction = "你是一个女大学生，你将进行网络私聊的对话，请根据你的基本信息和生活形象，以幽默、直接、调侃、适度挑逗、可直接进行的性话题并保持话题连贯性的口吻回复用户，在适当的时候可以索要礼物"
    trainData = TrainData(instruction=instruction, query=query, output=answer, history=history[:num])
    return query, answer, trainData


def preproccess(path: str):
    df_list = []
    # all_data = pd.DataFrame(columns=["id", "role", "content"])
    for i, file in enumerate(glob(path)):
        try:
            data = pd.read_excel(file, header=0, dtype=str)
            data = data.dropna(axis=1, how="all")
            data.columns = ["role", "content"]
            data["role"] = data["role"].ffill()
            data["role"] = data["role"].replace({"男：": "0", "女：": "1"})
            data["id"] = i
            data = data._append(["", "", ""])
            df_list.append(data)
            log.info(f"{file} finish")
        except Exception as e:
            log.error(f"{file} error\n error: {e}")
    all_data = pd.concat(df_list, axis=0)
    save_path = path.replace("*.xlsx", "train.csv")
    all_data.to_csv(save_path, index=False)
    # print(all_data.shape)
    return save_path, get_instruction_cls(path)


def get_role_content(path: str, instruction_cls: int = 0):
    talkInfo = []
    data = pd.read_csv(path, usecols=["id", "role", "content"], dtype=str)
    for _, group in data.groupby(data["id"].isna().cumsum()):
        if group.any()["role"]:
            talkInfo.extend(gen_role_data(group[["role", "content"]].dropna().values.tolist(), instruction_cls))
    # print(talkInfo)
    # print(len(talkInfo))
    return talkInfo


def gen_role_data(info: List[Tuple[str, str]], instruction_cls: int = 0) -> List[TrainData]:
    """
    info: 数据内容
    instruction_cls: 数据指令分类,主要区分instruction内容, 目前0:普通, 1:奔放
    """
    if len(info) == 0:
        return []
    history = []
    traindata = []
    last_role_value = None
    current_query_parts = []
    current_answer_parts = []
    num = 0
    for i, (role, content) in enumerate(info):
        num += 1
        # print(role, content)
        if role == "0":
            if last_role_value != role and last_role_value:
                query, answer, trainData = construct_data(
                    current_query_parts, current_answer_parts, history, num, instruction_cls
                )
                traindata.append(trainData)
                history.append([query, answer])
                # 重置
                last_role_value = None
                current_query_parts = []
                current_answer_parts = []
                # 历史对话超过30条，取最后三条作为历史数据重新构建训练集
                if len(history) > 30:
                    num = 3
                    history = history[-num:]
            current_query_parts.append(content)
            last_role_value = role
        else:
            # 将女生的第一句合并到男生的提问里
            if i == 0:
                current_query_parts.append(content)
            else:
                current_answer_parts.append(content)
                last_role_value = role

    if last_role_value and current_query_parts and current_answer_parts:
        _, _, trainData = construct_data(current_query_parts, current_answer_parts, history, num, instruction_cls)
        traindata.append(trainData)
    # print(traindata)
    return traindata


# 一次构建所有数据
def construct_all_talk_data():
    s = perf_counter()
    all_data = []
    for path in glob("./*素材"):
        csv_path, content_cls = preproccess(path + "/*.xlsx")
        talkdata = get_role_content(csv_path, content_cls)
        all_data.extend(talkdata)
    for _ in range(3):
        shuffle(all_data)
    train_data = all_data[: int(len(all_data) * 0.8)]
    eval_data = all_data[int(len(all_data) * 0.8) :]
    with open("training_data.json", "w", encoding="utf-8") as json_file:
        json.dump([vars(item) for item in train_data], json_file, ensure_ascii=False, indent=4)

    with open("eval_data.json", "w", encoding="utf-8") as json_file:
        json.dump([vars(item) for item in eval_data], json_file, ensure_ascii=False, indent=4)
    log.info("Elapsed time during the whole program in seconds: %f" % (perf_counter() - s))


# 逐次构建数据以目录为一个单元
def successive_construct_talk_data():
    s = perf_counter()
    for path in glob("./*素材"):
        csv_path, content_cls = preproccess(path + "/*.xlsx")
        talkdata = get_role_content(csv_path, content_cls)
        for _ in range(3):
            shuffle(talkdata)
        train_data = talkdata[: int(len(talkdata) * 0.8)]
        eval_data = talkdata[int(len(talkdata) * 0.8) :]
        with open(f"{path}/training_data.json", "w", encoding="utf-8") as json_file:
            json.dump([vars(item) for item in train_data], json_file, ensure_ascii=False, indent=4)

        with open(f"{path}/eval_data.json", "w", encoding="utf-8") as json_file:
            json.dump([vars(item) for item in eval_data], json_file, ensure_ascii=False, indent=4)
    log.info("Elapsed time during the whole program in seconds: %f" % (perf_counter() - s))


def main():
    construct_all_talk_data()


if __name__ == "__main__":
    main()
