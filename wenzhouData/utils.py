import re
from html import escape
from io import BytesIO

import pandas as pd
from docx import Document as Docx
from docx.document import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from wenzhouData.logger import log


class DocxProcess:
    def __init__(self):
        pass

    def __clean(self, line):
        line = re.sub(r"\u3000", " ", line).strip()
        return line

    def iter_block_items(self, parent):
        # 判断传入的是否为word文档对象，是则获取文档内容的全部子对象
        if isinstance(parent, Document):
            parent_elm = parent.element.body
        # 判断传入的是否为单元格，是则获取单元格内全部子对象
        elif isinstance(parent, _Cell):
            parent_elm = parent._tc
        else:
            raise ValueError("something's not right")

        # 遍历全部子对象
        for child in parent_elm.iterchildren():
            # 判断是否为段落，是则返回段落对象
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            # 判断是否为表格，是则返回表格对象
            if isinstance(child, CT_Tbl):
                yield Table(child, parent)

    def _table2html(self, table):
        html_table = ["<table>"]
        # 构建表头
        html_table.append("<tr>")
        for cell in table.rows[0].cells:
            html_table.append(f"<th>{escape(cell.text.strip())}</th>")
        html_table.append("</tr>")
        # 构建表格主体
        for row in table.rows[1:]:
            html_table.append("<tr>")
            for cell in row.cells:
                html_table.append(f"<td>{escape(cell.text.strip())}</td>")
            html_table.append("</tr>")
        html_table.append("</table>")
        return "".join(html_table)

    def _table2markdown(self, table):
        # 初始化Markdown表格的第一行
        markdown_table = ["|"]

        # 获取表格的列数
        num_cols = len(table.rows[0].cells)

        # 构建表头
        for col_idx in range(num_cols):
            cell = table.rows[0].cells[col_idx]
            markdown_table.append(f" {cell.text.strip()} |")
        markdown_table.append("\n|")

        # 构建分隔行
        for _ in range(num_cols):
            markdown_table.append(" --- |")
        markdown_table.append("\n")

        # 构建表格主体
        for row in table.rows[1:]:
            markdown_table.append("|")
            for cell in row.cells:
                markdown_table.append(f" {cell.text.strip()} |")
            markdown_table.append("\n")
        return "".join(markdown_table)

    def _table_process(self, tb, fileType="xml"):
        if fileType == "md":
            table = self._table2markdown(tb)
            return table
        html = self._table2html(tb)
        if fileType == "text":
            return re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", html)
        return html

    def __call__(self, filename, binary=None, from_page=0, to_page=100000, table_type="xml"):
        self.doc = Docx(filename) if not binary else Docx(BytesIO(binary))
        # 遍历word文档，最后调用函数没有返回值时停止遍历
        pn = 0
        lines = []
        for p in self.iter_block_items(self.doc):
            if pn > to_page:
                break
            if p.style.type == WD_STYLE_TYPE.TABLE:
                tab = self._table_process(p, table_type)
                lines.append(tab)
                log.info(tab)
                continue
            elif from_page <= pn < to_page and p.text.strip():
                log.info((self.__clean(p.text), p.style.name))
                lines.append((self.__clean(p.text), p.style.name))
            for run in p.runs:
                if "lastRenderedPageBreak" in run._element.xml:
                    pn += 1
                    continue
                if "w:br" in run._element.xml and 'type="page"' in run._element.xml:
                    pn += 1
        return lines


def read_lmdb(env, proj_id, prefix="requirement"):
    txn = env.begin(write=False)
    # samplenum = int(txn.get(b"samplenum").decode("utf-8"))
    # log.info("samplenum: %d" % samplenum)
    requirement = txn.get(("%s-%s" % (prefix, proj_id)).encode("utf-8"))
    if requirement:
        return requirement.decode("utf-8")
    return None


def export_lmdb(env, requirements, batch=10000, prefix="requirement"):
    txn = env.begin(write=True)
    process_idx = 0
    for proj_id, requirement in requirements:
        requirement_index = "%s-%s" % (prefix, proj_id)
        txn.put(requirement_index.encode("utf-8"), requirement.encode("utf-8"))
        if process_idx % batch == 0:
            # print('process idx: %d' % process_idx)
            log.info("process idx: %d" % process_idx)
            txn.commit()
            txn = env.begin(write=True)
        process_idx += 1
    if process_idx % batch != 0:
        log.info("process idx: %d" % process_idx)
        txn.commit()
        txn = env.begin(write=True)
    # print('process totalnum: %d' % process_idx)
    log.info("process totalnum: %d" % process_idx)
    txn.put("samplenum".encode("utf-8"), str(process_idx).encode("utf-8"))
    txn.commit()


def get_valuable_info_by_excel(excel_path, columns=["功能序号", "用例名称", "用例性质", "步骤", "预期结果"]):
    original_data = pd.read_excel(excel_path)
    if columns and set(columns).issubset(set(original_data.columns)):
        original_data = original_data[columns]
        original_data["步骤"] = original_data["步骤"].str.replace(" ", "")
    # 去重
    df_deduped = original_data.drop_duplicates(subset=["功能序号", "用例名称", "用例性质"])
    for row in df_deduped.iloc():
        yield row


def main():
    # docx = DocxProcess()
    from time import perf_counter

    s = perf_counter()
    columns = ["功能序号", "用例名称", "用例性质", "步骤", "预期结果"]
    # docx("wenzhouData/WZB_日间头寸管理系统业务需求说明书.docx", from_page=0, to_page=100000)
    for rows in get_valuable_info_by_excel("wenzhouData/温州银行系统测试案例-脱敏.xlsx", columns):
        print(rows["功能序号"])
        print(rows[columns[1:]].to_dict())
    print(perf_counter() - s)


if __name__ == "__main__":
    main()
