#!/usr/bin/env python3
"""Bootstrap 95% CI for referable sens/spec, severe recall, 0-4 QWK.
Modes per model: 'audit' (per-lesion->map->tier) or 'blackbox' (parse Final grade).
Usage: score_bootstrap.py <set: 150|full>"""
import json, re, sys, random
random.seed(20260625)
LES=["MA","HE","EX","SE"]; REFER={"Moderate","Mod-or-Severe-indeterminate","Severe"}
T2G={"No-DR":0,"Mild":1,"Moderate":2,"Mod-or-Severe-indeterminate":2,"Severe":3}
ROOT="/sda/zgh/stage1_5_experiment/preds"; LFD="/sda/zgh/LLaMA-Factory/data"
def pa(t):
    m=re.search(r'"present"\s*:\s*(true|false)',str(t))
    if m:return m.group(1)=="true"
    s=str(t).lower();m=re.search(r"\b(present|absent|yes|no)\b",s[:80]) or re.search(r"\b(present|absent|yes|no)\b",s)
    return None if not m else m.group(1) in("present","yes")
def grade(t):
    t=str(t)
    m=re.search(r'final\s*grade\b[^0-4]{0,15}([0-4])',t,re.I)      # "final grade is/:/of X"
    if not m: m=re.search(r'\bgrade\b[^0-4]{0,18}([0-4])',t,re.I)   # "grade this as 'X'"
    if not m: m=re.search(r'([0-4])[^0-9]{0,8}$',t.strip())          # 末尾数字兜底
    return int(m.group(1)) if m else None
def load_map():
    return json.load(open('/sda/zgh/stage1_5_experiment/data/stage2_grade_distribution.json'))['fitted_map(pattern->tier)']
FMAP=load_map()
def recs_audit(qf,pf):
    Q=[json.loads(l) for l in open(qf)];P=[json.loads(l).get('predict','') for l in open(pf)]
    img={}
    for q,p in zip(Q,P):
        m=q['meta'];d=img.setdefault(m['image'],{"present":set(),"cg":m['clinical_grade']})
        if pa(p):d['present'].add(m['lesion'])
    out=[]
    for k,d in img.items():
        tier=FMAP.get("".join(c for c in LES if c in d['present']),"No-DR")
        out.append((d['cg'], tier in REFER, T2G[tier]))
    return out
def recs_bb(qf,pf):
    Q=[json.loads(l) for l in open(qf)];P=[json.loads(l).get('predict','') for l in open(pf)]
    out=[]
    for q,p in zip(Q,P):
        g=grade(p); g=0 if g is None else g; cg=q['meta']['clinical_grade']
        out.append((cg, g>=2, g))
    return out
def metrics(recs):
    tp=fp=fn=tn=sr=st=0;ti=[];pj=[]
    for cg,pref,pg in recs:
        cr=cg>=2
        tp+=cr&pref;fn+=cr&(not pref);fp+=(not cr)&pref;tn+=(not cr)&(not pref)
        if cg>=3:st+=1;sr+=pref
        ti.append(cg);pj.append(pg)
    sens=tp/(tp+fn) if tp+fn else 0;spec=tn/(tn+fp) if tn+fp else 0;srec=sr/st if st else 0
    return sens,spec,srec,qwk(ti,pj)
def qwk(ti,pj,k=5):
    n=len(ti)
    if not n:return 0
    O=[[0]*k for _ in range(k)]
    for i,j in zip(ti,pj):O[min(i,k-1)][min(j,k-1)]+=1
    rt=[sum(O[i]) for i in range(k)];ct=[sum(O[i][j] for i in range(k)) for j in range(k)];nu=de=0
    for i in range(k):
        for j in range(k):
            w=((i-j)/(k-1))**2;nu+=w*O[i][j];de+=w*rt[i]*ct[j]/n
    return 1-nu/de if de else 0
def ci(recs,B=1000):
    n=len(recs);base=metrics(recs);dists=[[],[],[],[]]
    for _ in range(B):
        s=[recs[random.randrange(n)] for _ in range(n)]
        for i,v in enumerate(metrics(s)):dists[i].append(v)
    def pct(d):d=sorted(d);return d[int(0.025*B)],d[int(0.975*B)]
    return base,[pct(d) for d in dists]
SET=sys.argv[1] if len(sys.argv)>1 else "150"
if SET=="150":
    qa=f"{LFD}/messidor2_audit_queries_sft.jsonl"; qb=f"{LFD}/bbcot_messidor2_sft.jsonl"
    models=[("我们v3se-270(框架)","audit",f"{ROOT}/msd_v3se270.jsonl",qa),
            ("Qwen黑箱","blackbox",f"{ROOT}/bbcot_Qwen3-VL-8B-Instruct.jsonl",qb),
            ("InternVL黑箱","blackbox",f"{ROOT}/bbcot_InternVL3_5-8B-HF.jsonl",qb),
            ("Lingshu黑箱","blackbox",f"{ROOT}/bbcot_Lingshu-I-8B.jsonl",qb)]
elif SET=="full":
    qa=f"{LFD}/msd1744_audit_queries.jsonl"; qb=f"{LFD}/msd1744_blackbox.jsonl"
    models=[("我们v3se-270(框架)","audit",f"{ROOT}/msd1744_v3se270_audit.jsonl",qa),
            ("Qwen黑箱","blackbox",f"{ROOT}/msd1744_bb_Qwen.jsonl",qb),
            ("InternVL黑箱","blackbox",f"{ROOT}/msd1744_bb_InternVL.jsonl",qb),
            ("Lingshu黑箱","blackbox",f"{ROOT}/msd1744_bb_Lingshu.jsonl",qb)]
print(f"=== Bootstrap 95%CI (set={SET}, B=1000) ===")
print(f"{'model':22} | RefSens(CI) | RefSpec(CI) | 重病召回(CI) | QWK(CI)")
for name,mode,pf,qf in models:
    try:
        recs=recs_audit(qf,pf) if mode=="audit" else recs_bb(qf,pf)
        b,c=ci(recs)
        print(f"{name:22} | {b[0]:.2f}[{c[0][0]:.2f},{c[0][1]:.2f}] | {b[1]:.2f}[{c[1][0]:.2f},{c[1][1]:.2f}] | {b[2]:.2f}[{c[2][0]:.2f},{c[2][1]:.2f}] | {b[3]:.2f}[{c[3][0]:.2f},{c[3][1]:.2f}]")
    except Exception as e:
        print(f"{name}: ERR {e}")
