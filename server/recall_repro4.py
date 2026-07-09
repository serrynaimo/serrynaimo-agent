import re
# Two Christians. G has a travel/visit memory; B has a film memory.
mem = [
    {"id":"G_trav","person":"Christian Geissendoerfer","content":"Christian flying into Singapore July 13, staying with Thomas"},
    {"id":"G_bio","person":"Christian Geissendoerfer","content":"Friend from Munich, living in Saigon"},
    {"id":"B_film","person":"Christian Bale","content":"Christian Bale film night idea"},
    {"id":"B_visit","person":"Christian Bale","content":"Christian Bale visiting Singapore for a premiere"},
]
people = ["Christian Geissendoerfer","Christian Bale","Michael Fairweather"]
def people_matching(name):
    n=name.strip().lower().lstrip("@")
    if len(n)<2: return []
    exact=[p for p in people if p.lower()==n]
    if exact: return exact
    if len(n)<3: return []
    return [p for p in people if n in p.lower()]
def recall(keywords, person=None, limit=5):
    mn={}; profile_kw=set()
    for k in keywords:
        names=people_matching(k)
        if names:
            profile_kw.add(k)
            for nm in names: mn[nm.lower()]=nm
    if person:
        for nm in people_matching(person): mn[nm.lower()]=nm
    matched=set(mn)
    secondary=[k for k in keywords if k not in profile_kw]
    ss={t[:4] for k in secondary for t in re.findall(r"[\w']+",k.lower()) if len(t)>2}
    ph,oh,kwp=[],[],set()
    for rec in mem:
        rp=(rec.get('person') or '').lower()
        hay=f"{rec['content']} {rec.get('person') or ''}".lower()
        hs={w[:4] for w in re.findall(r"[\w']+",hay) if len(w)>2}
        score=len(ss&hs)
        if matched and rp in matched:
            ph.append((score,rp,rec['id']))
            if score: kwp.add(rp)
        elif score: oh.append((score,rec['id']))
    if len(matched)>1:
        if len(kwp)==1:
            matched=kwp; ph=[t for t in ph if t[1] in kwp]
        else:
            return {"candidates":sorted(mn.values())}
    ph.sort(key=lambda x:x[0],reverse=True); oh.sort(key=lambda x:x[0],reverse=True)
    recs=[i for _,_,i in ph]+[i for _,i in oh[:max(0,limit-len(ph))]]
    return {"memories":recs}
print("['Christian'] (no kw, 2 match)        :", recall(['Christian']))
print("['Christian','visiting'] (BOTH visit) :", recall(['Christian','visiting']))
print("['Christian','flying'] (only G)       :", recall(['Christian','flying']))
print("['Christian','film'] (only B)         :", recall(['Christian','film']))
print("['Christian','golf'] (neither)        :", recall(['Christian','golf']))
