"""文件职责：回归验证笔迹输入边界、标注隔离、结构导出和数学关系数据提取。"""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from inkmath.ink import validate_strokes,load_input,raster,trajectory,context_raster,read_inkml
from inkmath.layout import Node,make,latex,canonical,mathml_latex
from inkmath.relations import gold_edges,pair_features


class InputTests(unittest.TestCase):
    """验证异常坐标不会进入模型，坐标文件的标签不能泄漏到推理。"""
    def test_invalid_inputs(self):
        for obj in [[],[[]],[[[float('nan'),1]]],[[[float('inf'),1]]],[[[True,1]]],[[5]],[{'x':[1],'y':[]}],[[[1,2]]]*129]:
            with self.subTest(obj=str(obj)[:40]):
                with self.assertRaises(ValueError):validate_strokes(obj)
    def test_point_bound(self):
        with self.assertRaises(ValueError):validate_strokes([[[0,0]]*12001])
    def test_annotation_not_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json';p.write_text(json.dumps({'strokes':[[[0,0],[1,2]]],'latex':'DO_NOT_USE_THIS','symbols':[{'label':'fake'}]}));self.assertEqual(load_input(p),[[[0.,0.],[1.,2.]]])
    def test_xy_protocol(self):
        self.assertEqual(validate_strokes([{'x':[1,2],'y':[3,4]}]),[[[1.,3.],[2.,4.]]])
    def test_xml_entity_rejected(self):
        with self.assertRaises(ValueError):read_inkml(b'<!DOCTYPE ink><ink/>')
    def test_coordinate_invariance(self):
        a=[[[0,0],[10,20]],[[1,18],[9,1]]];b=[[[p[0]*3+100,p[1]*3-200] for p in s] for s in a]
        self.assertTrue(np.array_equal(raster(a)[0],raster(b)[0]));np.testing.assert_allclose(trajectory(a),trajectory(b),atol=1e-6)
    def test_context_differentiates_missing_stroke(self):
        ink=[[[0,10],[20,10]],[[10,0],[10,20]]]
        self.assertGreater(context_raster(ink,[0]).sum(),0);self.assertEqual(context_raster(ink,[0,1]).sum(),0)
    def test_extreme_dot_context_is_bounded(self):
        ink=[[[0,0]], [[-1e8,-1e8],[1e8,1e8]]];self.assertEqual(context_raster(ink,[0]).shape,(1,32,32))


class StructureTests(unittest.TestCase):
    """验证上下标、积分求和和嵌套结构的 LaTeX 契约，不把代数等价当作字面识别。"""
    def node(self,label):return Node(label,[0,0,1,1])
    def test_integral(self):
        n=make('subsup',[self.node(r'\int'),self.node('0'),self.node('1')]);self.assertEqual(latex(n),r'\int_{0}^{1}')
    def test_sum_fraction(self):
        a=make('subsup',[self.node(r'\sum'),self.node('n=1'),self.node(r'\infty')]);b=make('frac',[self.node('1'),make('sup',[self.node('n'),self.node('2')])]);self.assertEqual(canonical(latex(make('row',[a,b]))),r'\sum_{n=1}^{\infty}\frac{1}{n^{2}}')
    def test_canonical_preserves_structure(self):
        self.assertNotEqual(canonical(r'x^{2}'),canonical(r'x_{2}'));self.assertNotEqual(canonical(r'\frac{a}{b+c}'),canonical(r'\frac{a}{b}+c'))
    def test_mathml_function_normalization(self):
        self.assertEqual(mathml_latex('<math><mi>int</mi></math>'),r'\int')
    def test_mathml_relation_supervision(self):
        r={'symbols':[{'ref':'i'},{'ref':'a'},{'ref':'b'}],'mathml':'<math xmlns="http://www.w3.org/1998/Math/MathML"><msubsup><mo xml:id="i">int</mo><mn xml:id="a">0</mn><mn xml:id="b">1</mn></msubsup></math>'};edges,root=gold_edges(r);self.assertEqual(root,0);self.assertEqual(set(edges),{(0,1,3),(0,2,2)})
    def test_relation_translation_scale_invariance(self):
        a=[0,0,10,20];b=[12,-10,18,0];aa=[v*3+100 for v in a];bb=[v*3+100 for v in b];np.testing.assert_allclose(pair_features(a,b,20),pair_features(aa,bb,60),atol=1e-6)

if __name__=='__main__':unittest.main()
