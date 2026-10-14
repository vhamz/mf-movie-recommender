import json, numpy as np
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parent.parent
work=Path(__file__).parent
a=np.loadtxt(root/'u.data',dtype=np.int64)
splits=[[],[],[]]
for u in sorted(set(a[:,0])):
 rows=a[a[:,0]==u]; rows=rows[np.lexsort((rows[:,1],rows[:,3]))]
 n=len(rows); nt=int(.8*n); nv=int(.1*n)
 for dst,rs in zip(splits,[rows[:nt],rows[nt:nt+nv],rows[nt+nv:]]):dst.extend(rs)
train,val,test=[np.array(s) for s in splits]
movies={}
genres=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
for line in (root/'u.item').read_text(encoding='latin1').splitlines():
 f=line.split('|'); movies[int(f[0])]={'title':f[1],'genres':[g for g,x in zip(genres,f[6:24]) if x=='1']}
mu=train[:,2].mean();pool=set(train[:,1]);users=set(train[:,0]);seen={u:set(np.concatenate([train[train[:,0]==u,1],val[val[:,0]==u,1]])) for u in users}
result={'counts':{'train':len(train),'val':len(val),'test':len(test),'trainItems':len(pool)},'mean':mu,'sparsity':1-len(a)/(len(set(a[:,0]))*len(movies)),
 'unseen':{name:{'ratings':sum(i not in pool or u not in users for u,i,r,t in rows),'movies':sorted(set(rows[:,1])-pool),'users':sorted(set(rows[:,0])-users)} for name,rows in [('val',val),('test',test)]},
 'baseline':{name:float(np.sqrt(np.mean((rows[:,2]-mu)**2))) for name,rows in [('val',val),('test',test)]}}
if (work/'results.json').exists():
 js=json.loads((work/'results.json').read_text());w=js['weights'];p=np.array(w['users']).reshape(w['uShape']);q=np.array(w['items']).reshape(w['iShape'])
 def pred(rows):return np.clip(mu+(p[rows[:,0]]*q[rows[:,1]]).sum(1),1,5)
 result['mf']={name:float(np.sqrt(np.mean((pred(rows)-rows[:,2])**2))) for name,rows in [('val',val),('test',test)]}
 warm=test[np.array([i in pool and u in users for u,i,r,t in test])]
 result['warmOnly']={'mf':float(np.sqrt(np.mean((pred(warm)-warm[:,2])**2))),'baseline':float(np.sqrt(np.mean((mu-warm[:,2])**2))),'ratings':len(warm)}
 for name in ['val','test']:assert abs(result['mf'][name]-(js['restoredVal'] if name=='val' else js['testRMSE']))<2e-6
 per=[]
 for u in sorted(set(test[:,0])):
  relevant=set(test[(test[:,0]==u)&(test[:,2]>=4),1])&pool-seen.get(u,set())
  if not relevant:continue
  candidates=sorted(pool-seen[u]);scores=np.clip(mu+q[candidates]@p[u],1,5)
  ids=[candidates[k] for k in np.argsort(-scores,kind='stable')[:10]]
  per.append({'userId':int(u),'relevant':len(relevant),'hit':len(set(ids)&relevant)})
 result['recall']=float(np.mean([r['hit']/r['relevant'] for r in per]));assert abs(result['recall']-js['recall']['overallRecall'])<1e-10
 result['recallUsers']=len(per)
 result['manualExample']=per[0]
 r=test[0];result['dotExample']={'userId':int(r[0]),'movieId':int(r[1]),'rating':int(r[2]),'mu':mu,'products':(p[r[0]]*q[r[1]]).tolist(),'dot':float(p[r[0]]@q[r[1]]),'prediction':float(pred(test[:1])[0])}
 result['analysis']=[]
 for u in [1,405]:
  history=np.concatenate([train[train[:,0]==u],val[val[:,0]==u]])
  liked=history[history[:,2]>=4]; c=Counter(g for row in liked for g in movies[row[1]]['genres'])
  recs=next(x['recs'] for x in js['recommendations'] if x['userId']==u)
  for rec in recs:assert rec['movieId'] not in seen[u] and rec['movieId'] in pool
  result['analysis'].append({'userId':u,'historyCount':len(history),'likedCount':len(liked),'likedGenres':c.most_common(),
    'pastExamples':[{'movieId':int(row[1]),'rating':int(row[2]),**movies[row[1]]} for row in liked[:20]],
    'recs':[dict(r,genres=movies[r['movieId']]['genres']) for r in recs]})
encoded=json.dumps(result,indent=2,default=lambda x:x.item())
print(encoded)
(work/'verification.json').write_text(encoded)
