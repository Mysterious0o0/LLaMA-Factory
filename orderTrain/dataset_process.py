import ast
import json


temp_format = {
    "instruction": """请根据输入的目标、页面的元素信息和历史执行情况来输出手机下一步要执行什么操作，具体执行逻辑如下：
    1.解析输入的目标和页面元素信息综合历史操作信息判断下一步的操作逻辑
    2.每一步都判断是否达成目标。
        - 如果已达成目标，返回已达成目标。
        - 如果未达成目标并且存在可操作的元素，获取需要操作的元素坐标。返回点击操作或者输入操作
        - 如果未达成目标并且无法找到可操作的元素，获取滑动操作的起始位置坐标。返回滑动操作
    3.每次仅返回一步操作
    4.页面上出现问询的情况下，先对答复选项进行点击操作后，然后再继续任务
    5.历史操作信息中，如果步骤的状态是不正确，程序需要根据策略重新选择操作
    输入为一个字典格式如下:{target:\"\", points_info:[[text, text_type, [x1, y1, x2, y2]],...], history:[]}
    target:整个操作的目标
    points_info:OCR识别的页面元素，包括text:识别的元素文本, text_type:元素的类型, 以及[x1, y1, x2, y2]:元素的坐标
    history:之前为了target这个目的操作过的步骤
    返回结果有四种情况：
    1.需要进行点击操作：点击页面上的某一元素，则返回：{\"status\":\"not_completed\",\"action\":\"click\",\"position\":[x1,y1,x2,y2]
    ,\"content\":\"XXX\"}
    2.输入操作：在页面上进行文本输入，例如：{\"status\":\"not_completed\",\"action\":\"input\",\"position\":[x1,y1,x2,y2],\"conte
    t\":\"XXX\"}
    3.滑动操作：滑动页面，例如:{\"status\":\"not_completed\",\"action\":\"slide\",\"start\":[x1,y1],\"end\":[x2,y2]}将屏幕从start
    动到end，每次滑动的距离大于50小于1000
    4.已达成目标：{\"status\":\"completed\"}
    输出字段说明如下：
        - status:当前操作步骤完成与否，完成则返回\"completed\"，否则返回\"not_completed\"
        - action:预测的下一个步骤，可分为 1.click：点击point位置 2.input:在point的位置上输入text 3.slide:在从start滑动到end位置.
        - point:当action为click和input时，point是对应的元素坐标，当action是slide，则不返回此字段
        - text:根据具体action需要的其他信息，当action为click时，此字段为content代表其点击的文字信息，当action为input时，此字段为content代表需要输入的内容，当action为slide，则此字段为start和end，代表滑动的开始位置和结束位置
    返回逻辑：
    1. 解析输入的目标和页面元素信息综合历史操作信息判断下一步的操作逻辑
    2. 每一步都判断是否达成目标。
        - 如果已达成目标，返回已达成目标。
        - 如果未达成目标并且存在可操作的元素，获取需要操作的元素坐标。返回点击操作或者输入操作
        - 如果未达成目标并且无法找到可操作的元素，获取滑动操作的起始位置坐标。返回滑动操作
    3. 每次仅返回一步操作
    4. 操作建议包括：点击、输入、滑动
    5. 返回结果如果是点击操作并且history中的最后一步也是点击操作，需要判断两次点击的元素是否是同一个，如果是同一个元素请重新选择
    6. 返回结果为JSON字符串，不需要额外的解释说明信息，也不要写一段程序
    7. 当页面元素信息没有合适的元素时，优先滑动屏幕
    8. 页面上出现问询的情况下，先对答复选项进行点击操作后，然后再继续任务
    9. 返回结果是点击格式，那么返回的内容必须从输入的页面元素信息中选择一个。如返回结果{"status":"not_completed","action":"click","point":[x1,y1,x2,y2],"text":"XXX"},页面元素信息必须存在XXX(x1,y1,x2,y2)
    10. 历史操作信息中，如果步骤的状态是不正确，程序需要根据策略重新选择操作
    11. 如果target为滑动查找，那么目标元素优先选择type为text的元素""",
    "input": "",
    "output": "",
    "system": "手机智能代理",
    "history": [],
}


def data_process(data):
    question, answer = data["conversations"]
    question = question["value"]
    answer = answer["value"]
    input_text = question.split("suggest_actions(")[-1]
    index = input_text.index(",")
    target = input_text[:index]
    input_text = "(" + input_text[index + 1 :]
    ocr_result, history = ast.literal_eval(input_text)
    return target, ocr_result, history, answer


def main():
    data = []
    json_path = "/Users/cipher/Documents/work/longce/2024/data/dataset/train_baichuan0925.json"
    f = open(json_path, "r", encoding="utf-8")
    source_datas = json.loads(f.read())
    for source_data in source_datas:
        input_data = {}
        temp_format_n = temp_format.copy()
        target, ocr_result, history, answer = data_process(source_data)
        input_data["target"] = target
        input_data["points"] = ocr_result
        input_data["history"] = history
        temp_format_n["output"] = str(answer)
        temp_format_n["input"] = str(input_data)
        data.append(temp_format_n)
    with open("training_data1.json", "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
