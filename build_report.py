"""Create a compact HTML report and one-page PDF from measured artifacts.

Run with a Python containing reportlab; no model/training dependency required.
"""
from pathlib import Path
import html
import json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports"

def main():
    metadata = json.loads((ROOT/"artifacts/metadata.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT/"artifacts/evaluation.json").read_text(encoding="utf-8"))
    neighbors = json.loads((ROOT/"artifacts/neighbors.json").read_text(encoding="utf-8"))
    summary = evaluation["summary"]
    names = {"tfidf":"TF-IDF", "cbow":"TF-IDF × CBOW", "sg":"TF-IDF × Skip-Gram"}
    best_value = max(v["P@10"] for v in summary.values())
    leaders = [names[m] for m,v in summary.items() if abs(v["P@10"]-best_value)<1e-12]
    result = f'本次平均 P@10 最高為 {"、".join(leaders)}（{best_value:.3f}）。CBOW 的 P@10 高於 TF-IDF，但 MRR 較低，顯示前十名命中與首篇相關排名不一定同步改善。結果限於本次語料、參數與查詢，不能推論普遍優劣。'
    near = "、".join(f'{w}（{v:.2f}）' for w,v in neighbors["cbow"]["profit"])
    sections = [
        ("研究問題", "將 Word2Vec 的上下文資訊加入文件表示，能否比單純 TF-IDF 找到更相關的 Reuters 新聞？我分別檢查詞層級的鄰近關係，以及文件層級的搜尋排名，不把詞向量相似直接當成檢索改善。"),
        ("資料與方法", f'Reuters-21578 採完整 ModApte 分割：TRAIN {metadata["train_count"]:,} 篇、TEST {metadata["test_count"]:,} 篇。小寫化、英文字母斷詞、移除固定停用詞並保留否定詞；標題加內文，UNPROC 使用 TEXT。IDF 與詞向量只用 TRAIN 建立，全部 TEST 為候選。TF = 1 + ln(count)，IDF = 1 + ln((1 + N)/(1 + DF))；查詢與文件同樣處理，L2 正規化後計算 cosine。CBOW 以上下文預測中心詞，Skip-Gram 以中心詞預測上下文；Gensim 訓練 100 維、window 5、min_count 2、15 epochs、negative 5、seed 42、workers 1。文件向量採 TF-IDF 加權平均。'),
        ("評估設計", "事先固定 15 個英文查詢，earn、acq、crude、grain、trade 各 3 題。文件含目標 topic 即判為相關；P@10 為前 10 名命中比例，MRR 為首篇相關文件排名倒數的查詢平均，nDCG@10 使用二元相關性及排名折扣。以下為實測的查詢等權平均。"),
        ("結果與觀察", result + f' CBOW 的 profit Top-5 為：{near}。反義詞 loss 排在第一，正好說明向量相近不代表同義；swissair 的鄰近則可能反映特定公司財報語境，成因仍待回查原文。'),
        ("心得與限制", "我對 Semantic 的理解是：「語意相近，不代表詞彙關係相近」。更精確地說，向量反映的語意或情境接近，不保證同義、反義、上下位或可互換關係，也不保證符合相同的檢索意圖。profit 與 loss 共享財務上下文的可能性高，卻代表不同方向的結果。TF-IDF 加權平均也會失去詞序、否定與細節。因此，我不預設 Embedding 或 Skip-Gram 一定較好，而以實際排名驗證。topic 只能代理主題相關性，不能完整判斷使用者意圖；15 題、單一種子亦不足以支持普遍結論。")
    ]
    rows = [["方法", "P@10", "MRR", "nDCG@10"]] + [[names[m]]+[f'{summary[m][k]:.3f}' for k in ("P@10","MRR","nDCG@10")] for m in names]
    sources = '資料：Lewis，Reuters-21578，UCI，doi:10.24432/C52G6M。方法：Mikolov et al. (2013)，arXiv:1301.3781。'
    # HTML mirrors the PDF content and fits A4 when printing at 100%.
    content = ''
    for title, body in sections:
        content += f'<h2>{title}</h2><p>{html.escape(body)}</p>'
        if title == "評估設計":
            content += '<table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{html.escape(v)}</{"th" if i==0 else "td"}>' for v in row)+'</tr>' for i,row in enumerate(rows))+'</table>'
    markup = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>Reuters 新聞檢索實驗一頁報告</title>
<style>@page{{size:A4;margin:15mm}}*{{box-sizing:border-box}}body{{font-family:"Microsoft JhengHei",sans-serif;color:#182b43;line-height:1.65;background:#edf1f5;margin:0;font-size:10.5pt}}main{{width:210mm;min-height:297mm;margin:20px auto;background:white;padding:15mm}}h1{{color:black;font-size:21pt;margin:0 0 4mm}}h2{{color:black;font-size:12pt;margin:4mm 0 1mm}}p{{margin:0}}.meta{{font-size:9pt;color:#64748b;margin-bottom:5mm}}table{{width:100%;border-collapse:collapse;margin:3mm 0;font-size:10pt}}th{{background:#edf3f6;text-align:left}}td,th{{padding:2mm 3mm;border-bottom:1px solid #dde5ec}}footer{{font-size:8pt;color:#63748a;margin-top:4mm}}@media print{{body{{background:white}}main{{width:auto;min-height:0;margin:0;padding:0}}}}</style>
<main><h1>Reuters 新聞檢索實驗</h1><div class="meta">資訊檢索作業四　｜　姓名：____________　學號：____________<br>TF-IDF 與 TF-IDF 加權 Word2Vec 比較</div>{content}<footer>{sources}<br>展示形式：Streamlit 五段式報告，附即時搜尋、Top-5 鄰近詞、逐題指標與報告下載。</footer></main></html>'''
    (OUT/"one_page_report.html").write_text(markup,encoding="utf-8")
    font_path=Path("C:/Windows/Fonts/msjh.ttc")
    bold_path=Path("C:/Windows/Fonts/msjhbd.ttc")
    if not font_path.exists():
        raise RuntimeError("Set font_path/bold_path to available Traditional Chinese TTF/TTC fonts.")
    pdfmetrics.registerFont(TTFont("ReportCJK",str(font_path),subfontIndex=0))
    pdfmetrics.registerFont(TTFont("ReportCJKBold",str(bold_path),subfontIndex=0))
    width,height=595.276,841.89
    c=canvas.Canvas(str(OUT/"one_page_report.pdf"),pagesize=(width,height))
    c.setTitle("Reuters 新聞檢索實驗一頁報告")
    c.setAuthor("Information Retrieval Homework 4")
    left=43; usable=width-86; y=height-43
    normal=ParagraphStyle("body",fontName="ReportCJK",fontSize=10.3,leading=16,wordWrap="CJK",textColor=colors.HexColor("#1b2b3d"))
    small=ParagraphStyle("small",parent=normal,fontSize=7.4,leading=11,textColor=colors.HexColor("#596b80"))
    def para(text,style=normal):
        nonlocal y
        p=Paragraph(html.escape(text),style); _,h=p.wrap(usable,1000)
        p.drawOn(c,left,y-h); y-=h
    c.setFont("ReportCJKBold",21); c.drawString(left,y-21,"Reuters 新聞檢索實驗"); y-=36
    para("資訊檢索作業四　｜　姓名：____________　學號：____________",small)
    para("TF-IDF 與 TF-IDF 加權 Word2Vec 比較",small); y-=8
    for title, body in sections:
        y-=6; c.setFont("ReportCJKBold",11); c.setFillColor(colors.black); c.drawString(left,y-11,title); y-=18
        para(body)
        if title=="評估設計":
            y-=7
            t=Table(rows,colWidths=[usable*.48]+[usable*.52/3]*3)
            t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),"ReportCJK"),("FONTNAME",(0,0),(-1,0),"ReportCJKBold"),("FONTSIZE",(0,0),(-1,-1),9),("LEADING",(0,0),(-1,-1),12),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf3f6")),("LINEBELOW",(0,0),(-1,-1),.35,colors.HexColor("#dce4eb")),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5)]))
            _,h=t.wrap(usable,1000); t.drawOn(c,left,y-h); y-=h
    y-=11; para(sources,small)
    para("展示形式：Streamlit 五段式報告，附即時搜尋、Top-5 鄰近詞、逐題指標與報告下載。",small)
    if y<35: raise RuntimeError(f"Report overflows one-page budget: bottom={y}")
    c.showPage(); c.save()
    print(f"One-page PDF generated; bottom margin {y:.1f} pt")

if __name__=="__main__": main()
