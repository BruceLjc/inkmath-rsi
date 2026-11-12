"""文件职责：验证画板编辑使旧标签和旧异步结果失效，避免人工采样被错误标签污染。"""
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
from inkmath.draw import InkWindow


class DrawTests(unittest.TestCase):
    """离屏检查画板的版本与标签约束，不冒充真人窗口交互验收。"""
    def setUp(self):
        self.window=InkWindow(None)
    def tearDown(self):
        self.window.close();self.window.plt.close(self.window.fig)
    def test_new_ink_invalidates_example_label(self):
        w=self.window;w.example();self.assertTrue(w.label.text)
        w.press(SimpleNamespace(inaxes=w.ax,button=1,xdata=100,ydata=100));w.release(None)
        self.assertEqual(w.label.text,'');self.assertIsNone(w.last_result)
    def test_clear_invalidates_pending_result(self):
        w=self.window;w.example();old=w.revision;future=Future();w.future=future;w.clear()
        future.set_result({'revision':old,'latex':'SHOULD_NOT_DISPLAY'})
        w.poll();self.assertIsNone(w.last_result);self.assertEqual(w.label.text,'');self.assertEqual(w.strokes,[])
    def test_undo_invalidates_label(self):
        w=self.window;w.example();count=len(w.strokes);w.undo()
        self.assertEqual(len(w.strokes),count-1);self.assertEqual(w.label.text,'')

if __name__=='__main__':unittest.main()
