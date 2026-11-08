# Reuters 新聞檢索實驗

**系統網址**：[開啟 Reuters 新聞檢索系統](https://eriquechen23-cyl-reuters-ir-hw4-app-0ytgnp.streamlit.app/)

Streamlit 展示網頁與 A4 一頁報告。五段流程：研究問題、搜尋比較、詞向量觀察、實驗結果、心得與報告。每段有可關閉的口頭講稿提示。

## 本機執行

本專案已附 `artifacts/` 中的真實預訓練結果，不必重新下載或訓練。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

目前這台電腦已建立 `.venv`，可雙擊 `start_app.cmd` 啟動。

## Streamlit Community Cloud 部署

1. 將本專案上傳至自己的 GitHub 儲存庫。必須包含 `app.py`、`retrieval.py`、`queries.json`、`word_notes.json`、`requirements.txt`、`.streamlit/config.toml`、完整 `artifacts/` 與 `reports/`。
2. 登入 https://share.streamlit.io/ ，選擇 Create app，指定儲存庫與分支。
3. Main file path 填 `app.py`，Advanced settings 選 Python **3.12**。
4. Deploy 完成後，打開 `.streamlit.app` 網址，依序檢查五個頁籤及 PDF 下載。

不需要 API 金鑰、Secrets、GPU 或雲端訓練。不要上傳 `.venv/`、`data/raw/`、來源作業 DOCX 或本機測試暫存。雲端使用 `requirements.txt`，Gensim 僅供重新訓練使用。

官方說明：https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 重現實驗

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-train.txt
$env:PYTHONHASHSEED='42'
.venv\Scripts\python.exe prepare.py
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

官方來源 ZIP 下載約 8 MB。原始 SGML 按 `TOPICS="YES"` 且 `LEWISSPLIT="TRAIN"` / `"TEST"` 分為 9603 / 3299。無 TITLE/BODY 的 UNPROC 文件回退讀取 TEXT。保留整個官方分割，不依主題篩掉候選；詞彙表、DF、IDF、CBOW、Skip-Gram 皆只用 TRAIN 建立。未使用 TEST 訓練或調參。

英文小寫、正規式 `[a-z]+`、至少 2 字母、固定停用詞，保留 no/not/without；不做 stemming。TF 為 1+ln(count)，IDF 為 1+ln((1+N)/(1+DF))；以 L2 正規化後的內積計算 cosine。模型之外的詞被忽略；全未知詞、零向量不回傳排名。TF-IDF 僅回傳正重疊文件；有效 embedding 文件可有負 cosine。相同分數按 NEWID 排序。

CBOW / Skip-Gram 由 Gensim 4.4.0 訓練，100 維、window=5、min_count=2、15 epochs、5 negatives、seed=42、workers=1。上下文不跨文件但可能跨句。自製的是前處理、資料切分、TF/DF/IDF、文件表示、檢索與評估；不是手寫 Word2Vec 梯度更新器。單 worker 與 hash seed 改善重現性，跨平台浮點差異仍可能存在。

`queries.json` 於實驗前固定 15 題，五主題各三題。topic 標籤是主題相關性的代理，不是查詢特定人工標註。所有查詢等權平均：P@10 固定分母 10；MRR 搜尋完整有效排名；nDCG@10 使用二元相關性。未做多種子、顯著性檢定或超參數搜尋。

`artifacts/metadata.json` 記錄版本參數、資料雜湊、分割與訓練時間；`evaluation.json` 保存逐題指標與 Top-10；`manifest.json` 可核對輸出完整性。

## 報告

- `reports/one_page_report.pdf`：正式 A4 一頁報告，姓名學號留空待填。
- `reports/one_page_report.html`：同內容可列印版。
- `reports/evaluation.csv`：45 筆逐題 × 方法實測結果。
- `reports/presentation.md`：5–7 分鐘展示稿與常見問題。
- `docs/architecture.md`、`docs/reports/`：架構與驗證記錄。

PDF 重新產生使用 `build_report.py`，需要 ReportLab 及微軟正黑體；此步驟只在本機執行，雲端直接提供已產生的 PDF。重新訓練後須重新產生報告，避免數字不一致。

## 資料與方法引用

Lewis, D. Reuters-21578, Distribution 1.0. UCI Machine Learning Repository. https://doi.org/10.24432/C52G6M

原始資料 README 允許研究用途分發並要求標示來源；本專案用途為課程研究展示。新聞及原始標註著作權屬 Reuters Ltd.，資料相關說明保留於 `DATA_NOTICE.md`。

Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781

Gensim Word2Vec 文件：https://radimrehurek.com/gensim_4.0.0/models/word2vec.html
