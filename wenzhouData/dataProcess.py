import json
import os
import re
from glob import glob
from io import BytesIO
from random import shuffle
from time import perf_counter

import lmdb
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from tqdm import tqdm

from wenzhouData.utils import DocxProcess, export_lmdb, get_valuable_info_by_excel, log, read_lmdb


temp_format = {
    "instruction": """你是一个测试工程师，请根据输入的需求描述生成对应的'用例名称', '用例性质', '步骤', '预期结果',生成的内容以json的形式返回""",
    "input": "",
    "output": "",
    "system": "测试工程师",
    "history": [],
}


class Docx(DocxProcess):
    def __init__(self):
        pass

    def __clean(self, line):
        line = re.sub(r"\u3000", " ", line).strip()
        return line

    def __call__(self, filename, binary, from_page=0, to_page=100000, table_type="md"):
        doc = Document(filename) if not binary else Document(BytesIO(binary))
        pn = 0
        lines = []
        found = False
        for p in self.iter_block_items(doc):
            if pn > to_page:
                break
            if p.style.type == WD_STYLE_TYPE.TABLE:
                tab = self._table_process(p, table_type)
                if found:
                    lines.append(tab)
                continue
            elif from_page <= pn < to_page and p.text.strip():
                if "PROJ" in p.text:
                    if found:
                        yield lines
                        lines = []
                    else:
                        found = True
                if found:
                    if p.style.name == "Heading 3":
                        lines.append("### " + self.__clean(p.text) + "\n")
                    elif p.style.name == "Heading 4":
                        lines.append("#### " + self.__clean(p.text) + "\n")
                    elif p.style.name == "List Paragraph":
                        lines.append("- " + self.__clean(p.text) + "\n")
                    else:
                        lines.append(self.__clean(p.text) + "\n")
            for run in p.runs:
                if "lastRenderedPageBreak" in run._element.xml:
                    pn += 1
                    continue
                if "w:br" in run._element.xml and 'type="page"' in run._element.xml:
                    pn += 1


def get_docx_data(filename, binary=None, from_page=0, to_page=100000):
    requirements = []
    if re.search(r"\.docx$", filename, re.IGNORECASE):
        docx = Docx()
        for data in docx(filename, binary, from_page=from_page, to_page=to_page):
            pattern = r"\【PROJ(?:-\d+)+\】"
            match = re.search(pattern, data[0])
            if match:
                proj_id = match.group().strip("【】")
                data[0] = data[0].replace(match.group(), "")
                requirements.append((proj_id, "".join(data)))
                log.info(proj_id)
            else:
                log.warning("No match found for PROJ.")
    else:
        raise NotImplementedError("file type not supported yet(docx supported)")
    return requirements


def export_db():
    s = perf_counter()
    docx_path = "wenzhouData"
    dbfilename = "wenzhouData/dataDB"
    env = lmdb.open(dbfilename, map_size=1099511627776)
    for docx_file in tqdm(glob(os.path.join(docx_path, "*.docx"))):
        requirements = get_docx_data(docx_file, from_page=0, to_page=100000)
        export_lmdb(env, requirements)
        log.info(f"{docx_file} finish")
    env.close()
    log.info("Elapsed time during the whole program in seconds: %f" % (perf_counter() - s))


def get_train_eval_data():
    s = perf_counter()
    all_data = []
    dbfilename = "wenzhouData/dataDB"
    excel_path = "wenzhouData"
    excel_columns = ["功能序号", "用例名称", "用例性质", "步骤", "预期结果"]
    env = lmdb.open(dbfilename, map_size=1099511627776)
    for excel_file in tqdm(glob(os.path.join(excel_path, "*.xlsx"))):
        for rows in get_valuable_info_by_excel(excel_file, excel_columns):
            # print(rows["功能序号"])
            prod_id = rows["功能序号"]
            requirement = read_lmdb(env, prod_id)
            if not requirement:
                log.warning(f"No found PROJ: {prod_id}")
                continue
            output = rows[excel_columns[1:]].to_dict()
            temp_format["input"] = requirement
            temp_format["output"] = str(output).replace("\\n", "")
            all_data.append(temp_format)
    env.close()
    for _ in range(3):
        shuffle(all_data)
    train_data = all_data[: int(len(all_data) * 0.8)]
    eval_data = all_data[int(len(all_data) * 0.8) :]
    with open("wenzhouData/training_data.json", "w", encoding="utf-8") as json_file:
        json.dump(train_data, json_file, ensure_ascii=False, indent=4)

    with open("wenzhouData/eval_data.json", "w", encoding="utf-8") as json_file:
        json.dump(eval_data, json_file, ensure_ascii=False, indent=4)
    log.info("Elapsed time during the whole program in seconds: %f" % (perf_counter() - s))


if __name__ == "__main__":
    # export_db()
    get_train_eval_data()
