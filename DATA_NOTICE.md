# Reuters 資料來源與用途

本作業使用 **Reuters-21578, Distribution 1.0**，David D. Lewis 整理，原始新聞來自 Reuters 1987 年新聞資料。資料集與說明：https://archive.ics.uci.edu/dataset/137/reuters+21578+text+categorization+collection

DOI：https://doi.org/10.24432/C52G6M

原始 README 說明新聞文字及標註著作權屬 Reuters Ltd.，Reuters Ltd. 與 Carnegie Group, Inc. 允許研究用途的免費分發。本專案為資訊檢索課程研究展示。UCI 頁面另標示 CC BY 4.0；本專案保留原始研究用途說明，不擴張聲稱原始新聞的授權範圍。

`artifacts/documents.json` 包含 ModApte TEST 文件的標題、內容、topics 與 NEWID；數值模型與索引均由 TRAIN 學得。未將教師提供的作業 DOCX 包含在部署檔案中。
