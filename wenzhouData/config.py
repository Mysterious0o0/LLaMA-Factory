START1 = "proj"
START2 = "reqm"
END = "卍卍卍"
Heading3 = "Heading 3"
Heading4 = "Heading 4"
ListParagraph = "List Paragraph"
LastRenderedPageBreak = "lastRenderedPageBreak"
Xml1 = "w:br"
Xml2 = 'type="page"'

docx_path = "wenzhouData"
excel_path = "wenzhouData"
dbfilename = "wenzhouData/dataDB"

excel_columns = ["功能序号", "用例名称", "用例性质", "步骤", "预期结果"]

temp_format = {
    "instruction": """你是一个测试工程师，请根据输入的需求描述生成对应的'用例名称', '用例性质', '步骤', '预期结果',输入信息是markdown格式的描述文本,具体生成规则如下: 1. 生成正负用例样本 2. 生成的内容以json的形式返回""",
    "input": "",
    "output": "",
    "system": "测试工程师",
    "history": [],
}
