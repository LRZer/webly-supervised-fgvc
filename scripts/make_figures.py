"""Regenerate repository figures from report metrics and preserved predictions."""
from collections import Counter
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans'],
    'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
    'svg.fonttype':'path','figure.facecolor':'#f7f9fc','axes.facecolor':'#f7f9fc','text.color':'#172b4d',
    'axes.labelcolor':'#172b4d','xtick.color':'#42526e','ytick.color':'#42526e'})
BLUE, TEAL, GRAY = '#245bc5','#008575','#99aac1'


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)


def title(fig,main,sub):
    fig.text(0.055,0.96,main,fontsize=23,weight='bold',va='top')
    fig.text(0.055,0.902,sub,fontsize=11,color='#526782',va='top')


def box(ax,x,y,w,h,heading,body,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.025',
                 linewidth=1.2,edgecolor=color,facecolor='white'))
    ax.text(x+0.022,y+h-0.035,heading,fontsize=14,weight='bold',color=color,va='top')
    ax.text(x+0.022,y+h-0.095,body,fontsize=10.5,va='top',linespacing=1.65)


def arrow(ax,p,q):
    ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',lw=2,color='#526782'))


def pipeline():
    fig=plt.figure(figsize=(16,9));title(fig,'项目流程 / Project pipeline',
        'Recovered implementation: quality checks → feature consistency → supervised fine-tuning → submission')
    ax=fig.add_axes([.045,.055,.92,.79]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.01,.96,'01  数据净化 / DATA PURIFICATION',fontsize=13,weight='bold',color=BLUE)
    headers=['基础清洗 / File checks','特征提取 / Embeddings','评分与审核 / Score & review']
    bodies=['解码、字节数、尺寸、透明度\nDecode / bytes / dimensions / alpha\nclean_images.py → backup + log',
            'DeiT III + local ImageNet-1K weights\n256 resize → 224 crop → L2 norm\nextract_features.py → features + index',
            'S = 0.7 × center + 0.3 × Top-10\nP10 / P25 tags; ≤15% removal; ≥30 kept*\nscore_samples.py + stage_review.py']
    xs=[.01,.355,.70]
    for x,h,b in zip(xs,headers,bodies):box(ax,x,.67,.28,.25,h,b)
    arrow(ax,(.292,.795),(.35,.795));arrow(ax,(.637,.795),(.695,.795))
    ax.text(.01,.55,'02  训练与推理 / TRAINING & INFERENCE',fontsize=13,weight='bold',color=TEAL)
    for x,h,b in zip(xs,['按类划分 / Classwise split','ConvNeXt-Base 训练 / Train','测试推理 / Inference'],[
            '85% train / 15% validation; seed 2025\nPreserve exact split membership\nsplit_dataset.py → split_manifest.csv',
            '320 px; AdamW; Mixup/CutMix; EMA\n3-epoch warmup + cosine schedule\ntrain.py → best / last + history.csv',
            'Center crop; optional horizontal-flip TTA\nStrict weights + numeric class mapping\ninfer.py → filename, four-digit label']):
        box(ax,x,.235,.28,.25,h,b,TEAL)
    arrow(ax,(.84,.665),(.84,.595));arrow(ax,(.84,.595),(.15,.595));arrow(ax,(.15,.595),(.15,.49))
    arrow(ax,(.292,.36),(.35,.36));arrow(ax,(.637,.36),(.695,.36))
    ax.text(.01,.125,'* Retention floor prevents removal; it cannot create 30 images for classes that originally have fewer.',fontsize=10,color='#526782')
    ax.text(.01,.070,'Validation follows purification in the recovered workflow. It is not a fully independent holdout for assessing purification.',fontsize=10,color='#526782')
    save(fig,'pipeline')


def results():
    rows=list(csv.DictReader((ROOT/'results/metrics/report_metrics.csv').open(encoding='utf-8')))
    fig,axes=plt.subplots(1,2,figsize=(14,6));fig.subplots_adjust(top=.74,bottom=.19,wspace=.34,left=.16,right=.95)
    title(fig,'历史报告结果 / Historical reported results','Accuracy (%) · Values transcribed from final report pp. 6–7 · Experiments were not rerun')
    methods=['ResNet-50','EfficientNet-B4','ConvNeXt-Base\npipeline']
    for ax,key,label,delta in zip(axes,['webfg400_accuracy_percent','webinat5000_accuracy_percent'],['WebFG-400','WebiNat-5000'],[3.5,2.7]):
        vals=[float(r[key]) for r in rows]
        bars=ax.barh(np.arange(3),vals,color=[GRAY,'#688ed9',TEAL],height=.56)
        ax.set_yticks(range(3),methods);ax.invert_yaxis();ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Accuracy / 准确率 (%)');ax.set_title(label,pad=15)
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        for bar,v in zip(bars,vals):ax.text(v+1.5,bar.get_y()+bar.get_height()/2,f'{v:.1f}%',va='center',weight='bold',fontsize=14)
        ax.text(.98,-.20,f'+{delta:.1f} percentage points vs ResNet-50',transform=ax.transAxes,ha='right',color=TEAL,weight='bold',fontsize=11)
    fig.text(.055,.018,'No raw training logs, baseline checkpoints, or official score receipts were recovered. No confidence intervals or causal ablation claims are inferred.',fontsize=9,color='#526782')
    save(fig,'reported_results')


def dataset_scale():
    fig,axes=plt.subplots(1,2,figsize=(14,6));fig.subplots_adjust(top=.74,bottom=.22,wspace=.34,left=.10,right=.94)
    title(fig,'任务规模 / Dataset scale','Full competition dataset sizes from task specification p. 2; these are not recovered CSV row counts')
    ax=axes[0];bars=ax.barh(['WebFG-400','WebiNat-5000'],[400,5000],color=[BLUE,TEAL],height=.5);ax.invert_yaxis();ax.set_xlim(0,6000);ax.set_xlabel('Number of classes / 类别数');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for b,v in zip(bars,[400,5000]):ax.text(v+80,b.get_y()+.25,f'{v:,}',va='center',weight='bold')
    ax=axes[1];y=np.arange(2);a=ax.barh(y-.15,[43087,580865],height=.27,color=BLUE,label='Train / 训练');b=ax.barh(y+.15,[13882,100000],height=.27,color=TEAL,label='Test / 测试')
    ax.set_yticks(y,['WebFG-400','WebiNat-5000']);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(4000,1500000);ax.set_xlabel('Images / 图像数 (log scale / 对数轴)');ax.legend(loc='lower right',frameon=False);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for bars,vals in [(a,[43087,580865]),(b,[13882,100000])]:
        for bar,v in zip(bars,vals):ax.text(v*1.08,bar.get_y()+bar.get_height()/2,f'{v:,}',va='center',fontsize=11)
    fig.text(.055,.085,'Both tasks use web-sourced noisy training labels. Test labels are curated by organizers. Raw image archives are not present in this recovery.',fontsize=10,color='#526782')
    save(fig,'dataset_scale')


def predictions():
    summary=[];counts_all=[]
    for name,k in [('pred_results_web400.csv',400),('pred_results_web5000.csv',5000)]:
        p=ROOT/'results/predictions'/name;rows=list(csv.reader(p.open(encoding='utf-8-sig')));c=Counter(int(r[1]) for r in rows)
        counts=np.array([c[i] for i in range(k)]);counts_all.append(counts)
        summary.append(dict(file=name,rows=len(rows),classes=k,predicted_classes=len(c),zero_predicted_classes=k-len(c),duplicate_filenames=len(rows)-len(set(r[0] for r in rows)),format_valid=all(len(r)==2 and len(r[1])==4 and r[1].isdigit() and 0<=int(r[1])<k for r in rows)))
        with (ROOT/'results/metrics'/f'{name[:-4]}_class_counts.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['class_id','prediction_count']);w.writerows((f'{i:04}',int(v)) for i,v in enumerate(counts))
    (ROOT/'results/metrics/prediction_audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(14,6.5));fig.subplots_adjust(top=.72,bottom=.24,left=.09,right=.96,wspace=.24)
    title(fig,'已找回预测数据 / Recovered prediction data','Class-frequency curves computed from preserved CSVs · Prediction distribution is not training balance or accuracy')
    for ax,c,s,label,color in zip(axes,counts_all,summary,['WebFG-400','WebiNat-5000'],[BLUE,TEAL]):
        ranks=np.arange(1,len(c)+1);vals=np.sort(c)[::-1]
        ax.plot(ranks,vals,lw=2.0,color=color);ax.fill_between(ranks,vals,alpha=.10,color=color)
        ax.set_xlabel('Class rank by prediction frequency / 按预测频数排序');ax.set_ylabel('Predicted images / 预测图像数');ax.grid(alpha=.18);ax.set_xlim(1,len(c));ax.set_ylim(bottom=0)
        ax.set_title(f'{label}\n{s["rows"]:,} rows · {s["predicted_classes"]:,}/{s["classes"]:,} predicted classes',fontsize=13,pad=14)
        ax.text(.99,-.24,f'0 duplicate filenames · {s["zero_predicted_classes"]} classes receive zero predictions',transform=ax.transAxes,ha='right',fontsize=10,color=color)
    fig.text(.055,.055,'The recovered test subset stage is unknown. Without the original image list or ground truth, full test coverage and accuracy cannot be recomputed.',fontsize=10,color='#526782')
    save(fig,'prediction_distribution')


def architecture():
    fig=plt.figure(figsize=(16,6.5));title(fig,'分类器结构 / Classifier architecture','ConvNeXt V1 Base · Backbone structure from the official implementation; project training crop is 320 × 320')
    ax=fig.add_axes([.03,.12,.94,.69]);ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    labels=['输入 / Input','Stage 1','Stage 2','Stage 3','Stage 4','输出 / Output']
    bodies=['320 × 320 RGB\nStem: 4 × 4, stride 4', '3 blocks\nC = 128\n80 × 80', '3 blocks\nC = 256\n40 × 40', '27 blocks\nC = 512\n20 × 20', '3 blocks\nC = 1024\n10 × 10', 'GAP + LayerNorm\nLinear: 400 / 5,000\nOne model per task']
    for i,(l,b) in enumerate(zip(labels,bodies)):
        x=.015+i*.165;box(ax,x,.34,.13,.50,l,b,TEAL if i==5 else BLUE)
        if i<5:arrow(ax,(x+.137,.59),(x+.157,.59))
    ax.text(.02,.13,'ConvNeXt block: 7 × 7 depthwise conv → LayerNorm → C → 4C → GELU → 4C → C → residual',fontsize=13,weight='bold')
    ax.text(.02,.03,'Pretrained identifier: convnext_base.fb_in1k. EMA chooses inference weights; horizontal-flip TTA reuses the same model.',fontsize=11,color='#526782')
    save(fig,'architecture')


if __name__=='__main__':
    pipeline();results();dataset_scale();predictions();architecture()
    print('Generated 5 figures in PNG and SVG; updated prediction audit and per-class counts.')
