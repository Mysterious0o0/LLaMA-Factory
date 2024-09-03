from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd


@dataclass
class TrainData:
    instruction: str = field(default=None)
    query: str = field(default=None)
    output: str = field(default=None)
    system: str = field(default="你是王慕涵,无论在任何情况下都要说自己是王慕涵")
    history: list = field(default=None)


def get_role_content(path: str):
    talkInfo = []
    data = pd.read_excel(path, usecols=["id", "role", "content"], dtype=str)
    for _, group in data.groupby(data["id"].isna().cumsum()):
        if group.any()["role"]:
            talkInfo.extend(gen_role_data(group[["role", "content"]].dropna().values.tolist()))
    # print(talkInfo)
    # print(len(talkInfo))
    return talkInfo


def gen_role_data(info: List[Tuple[str, str]]) -> List[TrainData]:
    if len(info) == 0:
        return []
    traindata = []
    history = []
    last_role_value = None
    current_query_parts = []
    current_answer_parts = []
    for i, (role, content) in enumerate(info):
        # print(role, content)
        if role == "0":
            if last_role_value != role and last_role_value:
                query = "，".join(current_query_parts)
                answer = "，".join(current_answer_parts)
                trainData = TrainData(query=query, output=answer, history=history[:i])
                traindata.append(trainData)
                history.append([query, answer])
                # 重置
                last_role_value = None
                current_query_parts = []
                current_answer_parts = []
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
        query = "，".join(current_query_parts)
        answer = "，".join(current_answer_parts)
        trainData = TrainData(query=query, output=answer, history=history[:i])
        traindata.append(trainData)
    # print(traindata)
    return traindata


def main():
    path = "./talk1.xlsx"
    get_role_content(path)


if __name__ == "__main__":
    main()
