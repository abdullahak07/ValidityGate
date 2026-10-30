import json, numpy as np, sys
sys.argv=['x']
exec(open("analysis_figures.py").read().split("rng = np.random")[0])
rng=np.random.default_rng(1)
out={}
for thr_name,keep in [("RA>=0.70",lambda r,ref:r["RA"]>=0.70),
                      ("RA>=RA*-0.10",lambda r,ref:r["RA"]>=ref[1]-0.10),
                      ("RA>=0.70 & FA>=0.25",lambda r,ref:r["RA"]>=0.70 and r["FA"]>=0.25)]:
    print("\n==",thr_name)
    res={}
    for key in MAIN+EXTRA:
        bb,ds=key; ref=REF[key]; R=[r for r in rows(bb,ds) if keep(r,ref)]
        n_all=len(rows(bb,ds))
        if len(R)<8: print(f"{bb:14s}{ds:10s} n={len(R)}/{n_all} too few"); continue
        d=np.array([D(r,ref) for r in R]); FA=np.array([r["FA"] for r in R])
        line={"n":len(R),"n_all":n_all}
        for m in ["FA","1-RA","MIA","AD","JS"]:
            x=np.array([val(r,m) for r in R],float)
            (lo,hi),nc=boot(x,d,R,rng,2000)
            line[m]=dict(rho=float(spearmanr(x,d).correlation),lo=lo,hi=hi)
        line["n_cfg"]=nc
        for t in ["AD","JS"]:
            y=np.array([r[t] for r in R],float); (lo,hi),_=boot(FA,y,R,rng,2000)
            line["FA_vs_"+t]=dict(rho=float(spearmanr(FA,y).correlation),lo=lo,hi=hi)
        res[f"{bb}|{ds}"]=line
        print(f"{bb:14s}{ds:10s} n={len(R):3d}/{n_all:3d} cfg={nc:2d} " + " ".join(f"{m}:{line[m]['rho']:+.2f}[{line[m]['lo']:+.2f},{line[m]['hi']:+.2f}]" for m in ["FA","MIA","AD"]) + f" | FA~AD {line['FA_vs_AD']['rho']:+.2f} FA~JS {line['FA_vs_JS']['rho']:+.2f}")
    ks=[k for k in res if k.split("|")[0]!="LLaVA-shuffle"]
    pooled={m:meta([res[k][m]["rho"] for k in ks],[res[k]["n"] for k in ks]) for m in ["FA","1-RA","MIA","AD","JS"]}
    pooledc={m:meta([res[k][m]["rho"] for k in ks],[res[k]["n_cfg"] for k in ks]) for m in ["FA"]}
    for m,v in pooled.items(): print(f"  pooled {m:5s} {v['rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.4f} I2={v['I2']:.2f}")
    print(f"  pooled FA conservative {pooledc['FA']['rho']:+.2f} [{pooledc['FA']['lo']:+.2f},{pooledc['FA']['hi']:+.2f}] p={pooledc['FA']['p']:.3f}")
    out[thr_name]=dict(settings=res,meta=pooled,meta_cons=pooledc)
json.dump(out,open("stats_noncollapse.json","w"),indent=1)
