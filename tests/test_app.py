import unittest
from streamlit.testing.v1 import AppTest
from retrieval import ROOT

class AppTests(unittest.TestCase):
    def test_all_pages_and_edge_query(self):
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
        self.assertFalse(app.exception)
        for page in app.sidebar.radio[0].options:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(app.title)
            if page.startswith("02"):
                app.text_input[0].set_value("zzzxxyyqq").run()
                self.assertFalse(app.exception)
                self.assertGreaterEqual(len(app.warning),3)
            if page.startswith("03"):
                app.text_input[0].set_value("unknownzzzzz").run()
                self.assertFalse(app.exception)
                self.assertGreaterEqual(len(app.warning),2)

if __name__ == "__main__": unittest.main()
