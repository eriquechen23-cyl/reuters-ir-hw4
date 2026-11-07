"""A report-first Streamlit app for Reuters lexical and semantic retrieval."""
import html
import json
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from retrieval import ROOT, METHODS, Engine, tokenize, metrics

st.set_page_config(page_title="Reuters 語意檢索實驗", page_icon="🔎", layout="wide")
st.markdown("""<style>
.block-container{max-width:1280px;padding-top:4rem;padding-bottom:3rem}
h1{letter-spacing:-.035em} h2{letter-spacing:-.02em}
.eyebrow{font-size:.8rem;letter-spacing:.14em;color:#087f8c;font-weight:700;margin-bottom:.5rem}
.hero{background:#142b46;color:#fff;padding:30px 34px;border-radius:18px;margin:12px 0 24px}
.hero h2{color:#fff;margin:0;font-size:2rem}.hero p{color:#dae7f3;max-width:880px;margin-bottom:0}
.statement{border-left:5px solid #087f8c;background:#e9f4f3;padding:22px 26px;border-radius:0 12px 12px 0;font-size:1.3rem;margin:18px 0}
.news{background:white;border:1px solid #dfe6ee;border-radius:12px;padding:16px 18px;margin:10px 0}
.news h4{font-size:1rem;margin:7px 0}.news p{font-size:.88rem;color:#42566d;margin:8px 0;line-height:1.6}
.meta{font-size:.76rem;color:#587187}.tag{color:#087f8c;font-weight:600}
mark{background:#d2eeea;padding:0 2px} [data-testid="stMetric"]{background:white;border:1px solid #e0e7ef;padding:16px;border-radius:12px}
@media(max-width:700px){.hero{padding:22px}.hero h2{font-size:1.5rem}.block-container{padding:1rem}}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    return Engine()

if not (ROOT / "artifacts/metadata.json").exists():
    st.error("尚未建立實驗資料。請先執行 python prepare.py，再重新開啟網頁。")
    st.stop()
engine = load_engine()
evaluation = json.loads((ROOT / "artifacts/evaluation.json").read_text(encoding="utf-8"))
queries = json.loads((ROOT / "queries.json").read_text(encoding="utf-8"))
meta = engine.meta
TOPICS = {"earn":"企業獲利", "acq":"收購與合併", "crude":"原油", "grain":"穀物", "trade":"國際貿易"}
PAGES = ["01 研究問題", "02 搜尋比較", "03 詞向量觀察", "04 實驗結果", "05 心得與報告"]

with st.sidebar:
    st.markdown("### REUTERS / IR LAB")
    st.caption("Information Retrieval · Homework 4")
    page = st.radio("報告流程", PAGES, key="page")
    show_notes = st.toggle("顯示口頭報告提示", value=True)
    st.divider()
    st.caption("建議展示時間：5–7 分鐘")
    st.markdown("**報告主線**\n\n詞彙匹配 → 上下文表示 → 排名驗證 → 我的理解")
    st.caption("資料與模型已預先計算，切換頁面不需重新訓練。")

def note(text):
    if show_notes:
        st.info("🎤 你可以這樣說：" + text)

def heading(kicker, title, subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)

def result_table():
    return pd.DataFrame([{"方法":METHODS[m], **values} for m, values in evaluation["summary"].items()])

def highlighted(text, query):
    # Escape every source token before adding trusted highlight markup.
    import re
    words = set(tokenize(query))
    return "".join(f"<mark>{html.escape(p)}</mark>" if p.lower() in words else html.escape(p) for p in re.split(r"([A-Za-z]+)", text))

if page == PAGES[0]:
    heading("01 / QUESTION", "語意接近，搜尋就會更準嗎？", "Reuters-21578 · TF-IDF 與 TF-IDF 加權 Word2Vec 的檢索比較")
    st.markdown('<div class="hero"><h2>從「找相同的詞」到「找相近的上下文」</h2><p>同一個查詢，用三種文件表示搜尋同一批新聞。先觀察結果，再用指標判斷：Word Embedding 是否真的改善排名？</p></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, label, value in zip(cols, ["訓練新聞", "測試新聞", "固定評估查詢", "比較方法"], [f'{meta["train_count"]:,}', f'{meta["test_count"]:,}', evaluation["query_count"], 3]):
        col.metric(label, value)
    st.subheader("這次比較什麼")
    cols = st.columns(3)
    for col, title, body in zip(cols, ["A · TF-IDF", "B · TF-IDF × CBOW", "C · TF-IDF × Skip-Gram"], ["依詞項權重比對查詢與新聞。對確切詞彙敏感，無法自動連結不同字詞。", "用周圍詞預測中心詞，學到詞向量後，以 TF-IDF 加權表示整篇新聞。", "用中心詞預測周圍詞，同樣以 TF-IDF 加權形成文件向量。"]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**"); st.write(body)
    st.markdown("#### 實驗路徑")
    st.markdown("**官方 Reuters → ModApte 分割 → TRAIN 建立詞彙與模型 → TEST 新聞排名 → topic 相關性評估**")
    st.caption("所有可用 TEST 新聞都保留為候選；五個主題只用於設定查詢及判定相關性。")
    note("我的問題不是詞向量看起來合不合理，而是加入上下文表示後，相關新聞能不能真的排得更前面。")
    with st.expander("方法與公式：老師追問時展開"):
        st.markdown("英文轉小寫、保留長度至少 2 的字母詞、移除固定停用詞；保留 no / not / without，不做詞幹化。標題和內文各計入一次。")
        st.latex(r"TF(t,d)=1+\ln c(t,d),\quad DF(t)=|\{d\in TRAIN:t\in d\}|,\quad IDF(t)=1+\ln\frac{1+N_{train}}{1+DF(t)}")
        st.caption("未出現詞的 TF 設為 0；TF-IDF = TF × IDF，最後 L2 正規化。")
        st.latex(r"v_d=\frac{\sum_{t\in d\cap V_{W2V}} TFIDF(t,d)\,e_t}{\sum_{t\in d\cap V_{W2V}}TFIDF(t,d)},\quad score(q,d)=\frac{v_q^Tv_d}{\|v_q\|\|v_d\|}")
        st.write("程式先加權求和再 L2 正規化；與加權平均後計算 cosine 等價。查詢與文件使用相同權重。詞向量缺失時略過；全為未知詞時不回傳結果。")
        st.json(meta["params"])
        st.caption("TF、DF、IDF、Cosine 與評估公式自行實作；Word2Vec 最佳化使用 Gensim 4.4.0。固定 seed 42、單 worker；上下文不跨文件，但可能跨句。")

elif page == PAGES[1]:
    heading("02 / LIVE SEARCH", "同一個問題，三種搜尋答案", "先看前幾名新聞是否符合需求，再比較主題命中率。")
    presets = {f'{q["query"]} · {q["topic"]}':q for q in queries}
    choice = st.selectbox("選擇展示查詢", ["自訂查詢"] + list(presets), index=7)
    default = presets.get(choice, {"query":"profit loss", "topic":"earn"})
    query = st.text_input("輸入英文查詢", value=default["query"], key="query_"+choice, max_chars=300)
    topic = st.selectbox("用哪個主題檢查相關性（不影響排名）", ["不評估"] + list(TOPICS), index=list(TOPICS).index(default["topic"])+1, format_func=lambda x: f"{x} · {TOPICS[x]}" if x in TOPICS else x)
    count = st.slider("每種方法展示幾篇新聞", 3, 10, 3)
    if not tokenize(query):
        st.warning("請輸入至少一個有效英文詞，例如 crude oil。中文、數字或全停用詞不會產生可搜尋的詞項。")
    for col, (method, label) in zip(st.columns(3), METHODS.items()):
        with col:
            st.subheader(label)
            order, scores = engine.search(query, method)
            if topic != "不評估":
                values = metrics(order, engine.relevant(topic))
                st.metric("Precision@10", f'{values["P@10"]:.2f}')
                st.caption(f'MRR {values["MRR"]:.3f} · nDCG@10 {values["nDCG@10"]:.3f}')
            unknown = engine.unknown(query, method)
            if unknown:
                st.caption("未納入的詞：" + ", ".join(unknown))
            if not len(order):
                st.warning("沒有可用結果。請調整查詢；本系統不以零向量產生任意排名。")
            for position, i in enumerate(order[:count], 1):
                doc = engine.docs[int(i)]
                status = "主題符合" if topic in doc["topics"] else "主題不符" if topic != "不評估" else ""
                snippet = doc["body"][:240] + ("…" if len(doc["body"]) > 240 else "")
                st.markdown(f'<div class="news"><div class="meta">#{position:02d} · Reuters {doc["id"]} · cosine {scores[i]:.3f}</div><h4>{highlighted(doc["title"] or "(Untitled)",query)}</h4><div class="meta tag">{html.escape(", ".join(doc["topics"]))} · {status}</div><p>{highlighted(snippet,query)}</p></div>', unsafe_allow_html=True)
                with st.expander(f"閱讀 #{position} 全文", expanded=False):
                    st.write(doc["body"] or "此新聞沒有 BODY 內容。")
    st.caption("P@10 固定除以 10，即使畫面只展示 3 篇。TF-IDF 不回傳零重疊新聞；Embedding 可回傳負 cosine 的有效文件。不同方法的 cosine 尺度不能直接當成準確率比較。")
    note("我先用 crude oil prices 展示詞彙匹配，再換 petroleum production opec。排名改變只是觀察；是否改善，要看實際相關文件的排序。")

elif page == PAGES[2]:
    heading("03 / WORD LEVEL", "向量鄰居，不等於同義詞", "這裡觀察詞的上下文關係；上一頁與下一頁才檢查文件檢索。")
    word = st.text_input("觀察一個英文詞", value="profit", max_chars=60).strip().lower()
    if len(tokenize(word)) != 1 or tokenize(word)[0] != word:
        st.warning("請輸入單一有效英文詞，例如 profit、loss、oil 或 petroleum。")
    for col, method in zip(st.columns(2), ["cbow","sg"]):
        with col:
            st.subheader("CBOW" if method == "cbow" else "Skip-Gram")
            st.caption("周圍詞 → 平均詞向量 → 預測中心詞" if method == "cbow" else "中心詞 → 詞向量 → 預測周圍詞")
            neighbors = engine.neighbors(word, method)
            if neighbors:
                st.dataframe(pd.DataFrame(neighbors, columns=["Top-5 鄰近詞", "Cosine"]), hide_index=True, column_config={"Cosine":st.column_config.NumberColumn(format="%.3f")}, width="stretch")
            else:
                st.warning("這個詞不在詞向量詞彙表；訓練詞頻至少須為 2。")
    st.markdown('<div class="statement">語意相近，不代表詞彙關係相近。</div>', unsafe_allow_html=True)
    st.write("這裡的「語意相近」更精確地說，是模型學到的向量或上下文接近。它可能表示同一產業、相似句型或共同新聞背景，並不能單靠距離判定同義、反義、上下位或可互換關係。")
    st.write("例如 profit 與 loss 意思相反，仍可能出現在相似的財務敘述中；是否在本次模型接近，應以實際 Top-5 與 cosine 為準。oil 與 petroleum 則可用來觀察相關概念是否接近。")
    word_notes = json.loads((ROOT / "word_notes.json").read_text(encoding="utf-8"))
    if word in word_notes:
        st.info("本次實驗觀察：" + word_notes[word])
    with st.expander("完整 12 個觀察詞：Top-5 與解讀"):
        all_neighbors = json.loads((ROOT / "artifacts/neighbors.json").read_text(encoding="utf-8"))
        st.caption("以下為固定模型的實測鄰居。原因分析是語境解釋的假設，不是經過因果驗證的結論。")
        for probe, explanation in word_notes.items():
            st.markdown(f"**{probe}**")
            st.write("CBOW：" + "、".join(f"{w} ({s:.3f})" for w,s in all_neighbors["cbow"][probe]))
            st.write("Skip-Gram：" + "、".join(f"{w} ({s:.3f})" for w,s in all_neighbors["sg"][probe]))
            st.caption(explanation)
    with st.expander("從例句理解訓練流程"):
        st.code("例句：company reports higher profit this year\n中心詞：profit；視窗 2 → reports, higher, this, year\nCBOW：四個上下文詞的向量平均 → 預測 profit\nSkip-Gram：profit → 分別預測 reports / higher / this / year\nNegative sampling：提高正樣本分數，降低抽樣負詞分數，更新詞向量。")
        st.caption("上方是教學例句；實際前處理會移除 this 等停用詞，訓練視窗為 5。")
    with st.expander("PCA 二維投影：只輔助觀察"):
        method = st.selectbox("投影模型", ["cbow", "sg"], format_func=lambda x:METHODS[x])
        neighbors = engine.neighbors(word, method, 15)
        words = [word] + [w for w,_ in neighbors] if neighbors else []
        if len(words) >= 3:
            vectors = np.array([engine.embeddings[method][engine.vocab[w]] for w in words])
            centered = vectors - vectors.mean(axis=0)
            u,s,_ = np.linalg.svd(centered, full_matrices=False)
            xy = u[:,:2] * s[:2]
            frame = pd.DataFrame({"詞":words, "PC1":xy[:,0], "PC2":xy[:,1]})
            points = alt.Chart(frame).mark_circle(size=100,color="#087f8c").encode(x="PC1",y="PC2",tooltip=["詞","PC1","PC2"])
            labels = alt.Chart(frame).mark_text(dy=-12).encode(x="PC1",y="PC2",text="詞")
            st.altair_chart((points+labels).properties(height=330), width="stretch")
            st.caption(f"只對畫面中的 {len(words)} 個詞擬合 PCA；兩軸解釋變異 {(s[:2]**2).sum()/(s**2).sum():.1%}。原始 100 維 cosine 才是鄰近詞排序依據，二維距離有失真。")
    note("我理解的 Semantic 不能只看字面，也不能把向量接近直接當成同義詞。上下文關係、詞彙關係，以及搜尋需求是否相符，是三個需要分開檢查的問題。")

elif page == PAGES[3]:
    heading("04 / EVIDENCE", "把「看起來合理」交給數據驗證", "固定 15 個查詢、5 個主題，每個主題 3 題；所有方法使用完全相同的測試文件。")
    table = result_table()
    st.dataframe(table.style.format({"P@10":"{:.3f}","MRR":"{:.3f}","nDCG@10":"{:.3f}"}), hide_index=True, width="stretch")
    chart_data = table.melt(id_vars="方法", var_name="指標", value_name="分數")
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(x=alt.X("指標:N",title=None),xOffset="方法:N",y=alt.Y("分數:Q",scale=alt.Scale(domain=[0,1])),color=alt.Color("方法:N",scale=alt.Scale(range=["#142b46","#087f8c","#e7a54a"])),tooltip=["方法","指標",alt.Tooltip("分數:Q",format=".3f")]).properties(height=310)
    st.altair_chart(chart, width="stretch")
    best_value = max(v["P@10"] for v in evaluation["summary"].values())
    leaders = [METHODS[m] for m,v in evaluation["summary"].items() if np.isclose(v["P@10"],best_value,rtol=0,atol=1e-12)]
    st.success(f'本次固定查詢中，P@10 最高為 {"、".join(leaders)}（{best_value:.3f}）。這是此實驗設定下的結果。')
    st.warning("topic 標籤只是主題相關性的代理標準：同屬 earn 的虧損新聞，不一定符合『獲利增加』的查詢。15 個人工設計查詢、單一種子與小語料，不足以證明某方法普遍更好。")
    with st.expander("逐題評估與下載", expanded=True):
        selected_topic = st.selectbox("檢視主題", ["全部"] + list(TOPICS))
        rows = pd.DataFrame(evaluation["rows"])
        if selected_topic != "全部": rows = rows[rows.topic == selected_topic]
        rows = rows.copy(); rows["method"] = rows["method"].map(METHODS)
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button("下載完整評估 CSV", (ROOT / "reports/evaluation.csv").read_bytes(), "evaluation.csv", "text/csv")
    with st.expander("評估定義與可重現設定"):
        st.write("P@10：前 10 名主題相關文件數 / 10。MRR：每題第一篇主題相關文件排名的倒數，搜尋完整有效排名，再對 15 題平均。nDCG@10：二元 topic 相關性、折扣 1/log₂(rank+1)，除以理想前 10 名 DCG。表格均為查詢等權平均。")
        st.write(f'ModApte 完整分割 TRAIN {meta["train_raw"]:,} / TEST {meta["test_raw"]:,}。無 TITLE/BODY 的 UNPROC 新聞使用 TEXT 內容。全 TEST 候選，無主題預篩；無有效向量者不回傳。IDF、詞彙表與 Word2Vec 皆只擬合 TRAIN。')
        st.write("查詢於執行前固定於 queries.json；未依結果調整參數。相同分數依 Reuters NEWID 遞增排序。未作統計顯著性檢定，未跨種子重複實驗。")
        st.json(meta["params"])
    note("我同時比較詞與文件兩個層次。即使某些詞的鄰居很合理，整篇新聞取平均後仍可能丟失否定、詞序與細節，所以必須用文件排序指標驗證。")

else:
    heading("05 / TAKEAWAY", "我對 Semantic 的理解", "用一句主張、一個反例，再回到檢索效果，完成報告。")
    st.markdown('<div class="statement"><strong>語意相近，不代表詞彙關係相近。</strong></div>', unsafe_allow_html=True)
    st.markdown("我對 Semantic 的理解是：詞的意義需要放進上下文與使用目的中看。在 Word2Vec 裡，所謂接近主要來自上下文分布；它不會直接告訴我兩個詞是同義、反義、上下位，或能不能互換。因此，我會把這句話更精確地理解為：**向量所呈現的語意或情境接近，不代表詞彙具有相同關係，也不保證符合相同的檢索意圖。**")
    st.markdown("以 profit 和 loss 為例，它們都可能出現在財務新聞，卻代表不同方向的結果。若使用者想找獲利增加的新聞，只因上下文接近而推薦虧損新聞，未必能滿足需求。這也提醒我，topic 標籤只能驗證主題，仍需要人工閱讀才能判斷更細的意圖。")
    st.markdown("因此，我不會預設 Word Embedding 一定勝過 TF-IDF，也不會預設 Skip-Gram 一定勝過 CBOW。我會先觀察鄰近詞，再檢查文件排名與 P@10、MRR、nDCG@10。即使沒有全面改善，仍能幫助我理解詞彙匹配與上下文表示各自能保留、又會失去什麼。")
    st.subheader("帶走一頁式報告")
    c1,c2 = st.columns(2)
    pdf = ROOT / "reports/one_page_report.pdf"
    report_html = ROOT / "reports/one_page_report.html"
    if pdf.exists(): c1.download_button("下載 A4 一頁報告 PDF",pdf.read_bytes(),pdf.name,"application/pdf",type="primary")
    if report_html.exists(): c2.download_button("下載可列印 HTML 報告",report_html.read_bytes(),report_html.name,"text/html")
    with st.expander("5 分鐘口頭報告順序", expanded=True):
        st.markdown("1. **40 秒**：說明研究問題與三種方法。\n2. **90 秒**：用兩個預設查詢，比較前 3 篇新聞。\n3. **60 秒**：輸入 profit 或 oil，看兩種模型的鄰近詞。\n4. **70 秒**：讀平均指標與逐題結果，說明 topic 代理判斷的限制。\n5. **40 秒**：用自己的 Semantic 理解收尾。")
    note("這次實驗讓我把三件事分開：詞在向量空間是否接近、詞彙之間是什麼關係，以及文件是否符合使用者需求。真正的檢索效果，需要回到第三件事驗證。")
    st.caption("資料來源：David D. Lewis / Reuters-21578，UCI；模型訓練：Gensim Word2Vec。")
    st.markdown("[UCI 資料集](https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection) · [Word2Vec 原始論文](https://arxiv.org/abs/1301.3781) · [Streamlit 部署說明](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)")
