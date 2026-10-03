"""Check retention constraints, backups, deterministic splits, and CSV validity."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run(name,*args,success=True):
    p=subprocess.run([sys.executable,str(ROOT/'scripts'/f'{name}.py'),*map(str,args)],capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
    if success and p.returncode: raise AssertionError(p.stdout+p.stderr)
    if not success and not p.returncode: raise AssertionError('Expected rejection')
    return p


def read_rows(path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


class WorkflowTests(unittest.TestCase):
    def test_prediction_format_and_duplicates(self):
        m=module('validate_predictions')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pred.csv';p.write_text('a.jpg,0000\nb.jpg,0399\n',encoding='utf-8')
            self.assertEqual(m.validate(p,400,2)['rows'],2)
            for text in ['a.jpg,0000\na.jpg,0001\n','a.jpg,400\n','a.jpg,0400\n','../a.jpg,0000\n','a.jpg,0000,extra\n']:
                p.write_text(text,encoding='utf-8')
                with self.assertRaises(ValueError):m.validate(p,400)

    def test_cleaning_dry_run_and_backup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'input';root.mkdir();trash=Path(d)/'trash'
            Image.new('RGB',(80,80),'red').save(root/'good.png')
            Image.new('RGBA',(80,80),(0,0,0,0)).save(root/'transparent.png')
            Image.new('RGB',(12,12),'red').save(root/'tiny.png')
            (root/'bad.jpg').write_bytes(b'not an image')
            before={p.name:p.read_bytes() for p in root.iterdir()}
            opts=['--input',root,'--trash',trash,'--min-bytes',0,'--workers',1]
            run('clean_images',*opts)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            run('clean_images',*opts,'--apply')
            self.assertEqual([p.name for p in root.iterdir()],['good.png'])
            for name in ['transparent.png','tiny.png','bad.jpg']:
                self.assertEqual((trash/name).read_bytes(),before[name])

    def test_score_retention_and_small_classes(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);rng=np.random.default_rng(42)
            f=np.vstack([rng.normal([3,0,0],.1,size=(40,3)),rng.normal([0,3,0],.1,size=(4,3))]).astype('float32')
            f[0]=[-3,0,0];np.save(d/'features.npy',f)
            with (d/'index.csv').open('w',newline='',encoding='utf-8') as h:
                w=csv.writer(h);w.writerow(['path','class_id']);w.writerows((f'image_{i}.jpg',0 if i<40 else 1) for i in range(44))
            run('score_samples','--feat',d/'features.npy','--index',d/'index.csv','--outdir',d/'scores')
            kept=read_rows(d/'scores/kept_list.csv')
            removed=read_rows(d/'scores/removed_list.csv')
            self.assertEqual(len(kept)+len(removed),44)
            self.assertTrue(0<len(removed)<=6)
            self.assertTrue(all(r['class_id']=='0' for r in removed))
            self.assertEqual(sum(r['class_id']=='1' for r in kept),4)
            self.assertIn('image_0.jpg',{r['path'] for r in removed})

    def test_split_membership_repeats(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);src=d/'src'
            for c,n in [('0000',10),('0001',1)]:
                (src/c).mkdir(parents=True)
                for i in range(n):Image.new('RGB',(64,64),'blue').save(src/c/f'{i}.png')
            for n in ['a','b']:run('split_dataset','--src',src,'--dst',d/n,'--workers',1)
            a=read_rows(d/'a/split_manifest.csv')
            b=read_rows(d/'b/split_manifest.csv')
            self.assertEqual(a,b);self.assertEqual(len(a),11)
            self.assertEqual(len({(r['class_name'],r['filename']) for r in a}),11)
            self.assertEqual(next(r['split'] for r in a if r['class_name']=='0001'),'train')
            run('split_dataset','--src',src,'--dst',d/'a',success=False)

    def test_review_rejects_outside_root(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);root=d/'images';root.mkdir();outside=d/'outside.jpg';outside.write_bytes(b'protected')
            for name in ['removed','scores']:
                with (d/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.writer(f);w.writerow(['path','class_id']);w.writerow([str(outside),0])
            run('stage_review','--root',root,'--removed',d/'removed.csv','--score-csv',d/'scores.csv','--review-dir',d/'review','--delete','--force-delete',success=False)
            self.assertEqual(outside.read_bytes(),b'protected')

    def test_preserved_artifacts(self):
        import hashlib
        entries=json.loads((ROOT/'results/artifact_manifest.json').read_text(encoding='utf-8'))
        for item in entries:
            p=ROOT/item['path']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),item['sha256'])
        m=module('validate_predictions')
        m.validate(ROOT/'results/predictions/pred_results_web400.csv',400,5687)
        m.validate(ROOT/'results/predictions/pred_results_web5000.csv',5000,60000)


if __name__=='__main__':unittest.main()
