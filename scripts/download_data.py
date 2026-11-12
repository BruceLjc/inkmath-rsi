"""文件职责：下载并校验本轮研究用 CROHME 坐标归档；不执行归档内的第三方代码。"""
import hashlib,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
URL='https://raw.githubusercontent.com/ThomasLech/CROHME_extractor/master/data/CROHME_full_v2.zip'
EXPECTED='f9fcbaece8410c76c7fe4c47c15702f0d5e7aa572737805b2e2af596e7de7f75'
def main():
    """只保存原始归档并验证完整性；归档带非商用条款，产品训练须另行授权。"""
    p=ROOT/'data/raw/CROHME_full_v2.zip';p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        temp=p.with_suffix('.download');urllib.request.urlretrieve(URL,temp);temp.replace(p)
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if digest!=EXPECTED:raise RuntimeError('数据归档校验失败，请核对来源；未执行解包或训练')
    print('已校验研究数据：CC BY-NC-SA 3.0；不作为商业产品训练授权。')
if __name__=='__main__':main()
