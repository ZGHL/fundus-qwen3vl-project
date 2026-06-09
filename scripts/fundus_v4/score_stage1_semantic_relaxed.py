#!/usr/bin/env python3
"""Score lesion presence while allowing reasonable base-model JSON key variants."""
from __future__ import annotations
import argparse,json,re
from collections import Counter,defaultdict
from pathlib import Path
LESIONS=('MA','HE','EX','SE')

def obj(text):
 m=re.search(r'\{.*\}',text,re.S)
 if not m:return None
 try:return json.loads(m.group(0))
 except:return None

def bool_value(o):
 if not isinstance(o,dict):return None
 for k in ('present','target_lesion_present','lesion_present'):
  if isinstance(o.get(k),bool):return o[k]
 return None

def lesion(o):
 if not isinstance(o,dict):return None
 v=o.get('target_lesion') or o.get('lesion')
 if isinstance(v,dict):v=v.get('abbreviation') or v.get('name')
 s=str(v or '').upper()
 names={'MICROANEURYSM':'MA','RETINAL HEMORRHAGE':'HE','HEMORRHAGE':'HE','HARD EXUDATE':'EX','SOFT EXUDATE':'SE'}
 return s if s in LESIONS else names.get(s)

def met(c):
 tp,fp,fn,tn=[c[x] for x in ('tp','fp','fn','tn')]; div=lambda a,b:a/b if b else 0.0
 r=div(tp,tp+fn); s=div(tn,tn+fp)
 return {'n':tp+fp+fn+tn,'positive':tp+fn,'negative':tn+fp,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':div(tp,tp+fp),'recall':r,'specificity':s,'f1':div(2*tp,2*tp+fp+fn),'balanced_accuracy':(r+s)/2}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('predictions',type=Path); ap.add_argument('--json-out',type=Path,required=True); a=ap.parse_args()
 cs=defaultdict(Counter); total=Counter()
 for line in a.predictions.open(encoding='utf-8'):
  row=json.loads(line); po,lo=obj(str(row.get('predict',''))),obj(str(row.get('label',''))); l=lesion(lo); g=bool_value(lo); q=bool_value(po)
  total['n']+=1; total['json']+=po is not None; total['semantic']+=q is not None
  if l not in LESIONS or g is None:continue
  if q is None:cs[l]['fn' if g else 'fp']+=1
  elif q and g:cs[l]['tp']+=1
  elif q and not g:cs[l]['fp']+=1
  elif not q and g:cs[l]['fn']+=1
  else:cs[l]['tn']+=1
 by={x:met(cs[x]) for x in LESIONS}; keys=('f1','recall','specificity','balanced_accuracy')
 out={'n':total['n'],'json_parse_success':total['json']/total['n'],'semantic_present_success':total['semantic']/total['n'],'main4_macro':{k:sum(by[x][k] for x in LESIONS)/4 for k in keys},'by_lesion':by}
 a.json_out.parent.mkdir(parents=True,exist_ok=True); a.json_out.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
