import hashlib
import json
import unittest
import numpy as np
from retrieval import ROOT, Engine, fit_tfidf, transform, metrics, tokenize, METHODS
from prepare import parse_sgml

class FormulaTests(unittest.TestCase):
    def test_df_idf_and_cosine_by_hand(self):
        vocab, df, idf = fit_tfidf([["oil","oil","price"],["oil","grain"]])
        self.assertEqual(df[vocab["oil"]], 2)
        self.assertAlmostEqual(idf[vocab["oil"]], 1.)
        self.assertAlmostEqual(idf[vocab["grain"]], 1+np.log(3/2), places=6)
        matrix = transform([["oil","oil","price"],["grain"],[]],vocab,idf)
        a = 1+np.log(2); b = 1+np.log(3/2)
        self.assertAlmostEqual(matrix[0,vocab["oil"]],a/np.sqrt(a*a+b*b),places=6)
        self.assertAlmostEqual(float((matrix[0]@matrix[0].T).toarray()[0,0]),1.,places=6)
        self.assertEqual(matrix[2].nnz, 0)

    def test_known_metrics(self):
        m = metrics(np.array([0,1,2]),{1,2})
        self.assertEqual(m["P@10"],.2)
        self.assertEqual(m["MRR"],.5)
        self.assertAlmostEqual(m["nDCG@10"],(1/np.log2(3)+.5)/(1+1/np.log2(3)))
        self.assertTrue(all(v==0 for v in metrics([],set()).values()))

    def test_modapte_filter(self):
        def row(n, topic, split):
            return f'<REUTERS NEWID="{n}" TOPICS="{topic}" LEWISSPLIT="{split}"><TOPICS><D>earn</D></TOPICS><TITLE>A &amp; B</TITLE><BODY>profit</BODY></REUTERS>'
        docs = parse_sgml(row(1,"YES","TRAIN")+row(2,"YES","TEST")+row(3,"NO","TRAIN")+row(4,"BYPASS","TEST")+row(5,"YES","NOT-USED"))
        self.assertEqual([d["id"] for d in docs],[1,2])
        self.assertEqual(docs[0]["title"],"A & B")
        self.assertIn("not",tokenize("not profit"))
        raw='<REUTERS NEWID="9" TOPICS="YES" LEWISSPLIT="TEST"><TEXT TYPE="UNPROC">Reserve report</TEXT></REUTERS>'
        self.assertEqual(parse_sgml(raw)[0]["body"],"Reserve report")

@unittest.skipUnless((ROOT / "artifacts/manifest.json").exists(),"requires prepared real data")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.engine=Engine()

    def test_no_leakage_and_artifact_integrity(self):
        e=self.engine
        self.assertFalse(set(e.meta["train_ids"]) & {d["id"] for d in e.docs})
        self.assertTrue(all(d["split"]=="TEST" for d in e.docs))
        manifest=json.loads((ROOT/"artifacts/manifest.json").read_text())
        for name,digest in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT/"artifacts"/name).read_bytes()).hexdigest(),digest,name)

    def test_oov_and_blank_queries(self):
        for m in METHODS:
            for q in ["", "中文查詢", "the and of", "zzzxxyyqq"]:
                self.assertEqual(len(self.engine.search(q,m)[0]),0)

    def test_saved_metrics_are_reproducible(self):
        rows=json.loads((ROOT/"artifacts/evaluation.json").read_text())["rows"]
        for row in rows:
            ranking,_=self.engine.search(row["query"],row["method"])
            actual=metrics(ranking,self.engine.relevant(row["topic"]))
            for key,value in actual.items(): self.assertAlmostEqual(row[key],value)

    def test_neighbors_exclude_self(self):
        for m in ["cbow","sg"]:
            near=self.engine.neighbors("profit",m)
            self.assertEqual(len(near),5)
            self.assertNotIn("profit",[w for w,_ in near])

if __name__ == "__main__": unittest.main()
