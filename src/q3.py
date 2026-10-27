#!/usr/bin/env python3
"""Q3: train-only conditional generation and constrained Gaussian transport.

Run from repository root with PYTHONPATH=.pylibs:src and MPLBACKEND=Agg.
Default: five repeated within-style holdouts; final all-data spectra are descriptive.
T2N units are unresolved: retain both attachment-mg/L and alternative-ug/L scenarios.
"""
from __future__ import annotations
import argparse,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,norm,pearsonr,spearmanr
from scipy.spatial.distance import pdist,cdist
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,balanced_accuracy_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from data_utils import load_all,provenance,TARGET_CN

ROOT=Path(__file__).resolve().parents[1];OUT_DIR=ROOT/'data/q3';IMG_DIR=ROOT/'img'
SEED=42;TARGET_STYLES=['Tripel','Kriek','Lambic'];SOURCE_STYLE='Pils/Lager';TARGET_STYLE='Tripel'
N0=20;N_GEN=300
MODELS=['empirical','mean_noise','global_cov','copula','covmatch','ot','ot_rank','ot_constrained']
DEFECTS={'DMS':('DMS',17.25),'Diacetyl':('X2.3.butanedione',.09),
         '4VG':('X4VG',.10),'T2N':('X.E..2.nonenal',.0001)}
UNIT_SCENARIOS={'attachment_mg_L':.0001,'alternative_ug_L':.1}


def ensure_pd(cov,eps=1e-8):
    cov=(np.asarray(cov)+np.asarray(cov).T)/2
    w=np.linalg.eigvalsh(cov)
    return cov+np.eye(len(cov))*max(0,eps-float(w.min()))


def sqrt_psd(cov,inverse=False):
    w,v=np.linalg.eigh(ensure_pd(cov));w=np.maximum(w,1e-8)
    return (v*(1/np.sqrt(w) if inverse else np.sqrt(w)))@v.T


def gaussian_mapping(source_cov,target_cov,optimal=True):
    if not optimal:return sqrt_psd(target_cov)@sqrt_psd(source_cov,True)
    s=sqrt_psd(source_cov);inv=sqrt_psd(source_cov,True)
    return inv@sqrt_psd(s@target_cov@s)@inv


class ConcentrationTransform:
    """Training median imputation, dimensionless log1p(c/c0), training z-score."""
    def fit(self,X):
        X=np.asarray(X,float);self.median=np.nanmedian(X,axis=0)
        x=np.clip(np.where(np.isnan(X),self.median,X),0,None)
        self.c0=np.array([np.median(c[c>0]) if np.any(c>0) else 1. for c in x.T])
        z=np.log1p(x/self.c0);self.mean=z.mean(axis=0);self.sd=np.maximum(z.std(axis=0),1e-8)
        return self
    def clean(self,X):return np.clip(np.where(np.isnan(X),self.median,X),0,None)
    def transform(self,X):return (np.log1p(self.clean(X)/self.c0)-self.mean)/self.sd
    def inverse(self,Z):return self.c0*np.expm1(np.clip(Z*self.sd+self.mean,-30,30))


def normal_scores(x):
    return norm.ppf(np.clip((rankdata(x,axis=0,method='average')-.5)/len(x),1e-5,1-1e-5))


def corr_from_samples(z):
    cov=LedoitWolf().fit(z).covariance_;d=np.sqrt(np.maximum(np.diag(cov),1e-8))
    c=ensure_pd(cov/np.outer(d,d));d=np.sqrt(np.diag(c));return c/np.outer(d,d)


class FlavorGenerator:
    def __init__(self,n0=20):self.n0=n0
    def fit(self,X,styles,features):
        self.features=list(features);self.styles=np.asarray(styles)
        self.pre=ConcentrationTransform().fit(X);self.x=self.pre.clean(X);self.z=self.pre.transform(X)
        self.groups=sorted(set(styles));self.means={g:self.z[self.styles==g].mean(axis=0) for g in self.groups}
        residual=np.vstack([self.z[self.styles==g]-self.means[g] for g in self.groups])
        self.within_cov=ensure_pd(LedoitWolf().fit(residual).covariance_)
        us=np.vstack([normal_scores(self.x[self.styles==g]) for g in self.groups])
        self.within_corr=corr_from_samples(us)
        self.params={}
        for g in self.groups:
            xs=self.x[self.styles==g];zs=self.z[self.styles==g];a=self.n0/(len(xs)+self.n0)
            cov=ensure_pd((1-a)*LedoitWolf().fit(zs).covariance_+a*self.within_cov)
            corr=ensure_pd((1-a)*corr_from_samples(normal_scores(xs))+a*self.within_corr)
            d=np.sqrt(np.diag(corr));corr=corr/np.outer(d,d)
            self.params[g]={'mu':zs.mean(axis=0),'cov':cov,'corr':corr,'n':len(xs)}
        return self
    def caps(self,style,scenario):
        x=self.x[self.styles==style]
        strict={name:tau for name,(col,tau) in DEFECTS.items()};strict['T2N']=UNIT_SCENARIOS[scenario]
        return {col:max(strict[name],float(np.quantile(x[:,self.features.index(col)],.75)))
                for name,(col,_) in DEFECTS.items()}
    def generate(self,style,n=300,seed=42):
        rng=np.random.default_rng(seed);d=len(self.features);p=self.params[style]
        xs=self.x[self.styles==style];eps=rng.normal(size=(n,d))
        pool=np.vstack([self.z[self.styles==g]-self.means[g] for g in self.groups if g!=style])
        pool_mu=pool.mean(axis=0);pcov=ensure_pd(LedoitWolf().fit(pool).covariance_)
        raw=pool[rng.integers(0,len(pool),n)]-pool_mu
        independent_cov=np.diag(np.diag(p['cov']))
        out={'empirical':xs[rng.integers(0,len(xs),n)],
             'mean_noise':np.clip(self.pre.inverse(p['mu']+eps@sqrt_psd(independent_cov).T),0,None),
             'global_cov':np.clip(self.pre.inverse(p['mu']+eps@sqrt_psd(self.within_cov).T),0,None)}
        latent=eps@sqrt_psd(p['corr']).T;u=norm.cdf(latent)
        out['copula']=np.column_stack([np.quantile(xs[:,j],u[:,j]) for j in range(d)])
        for name,opt in [('covmatch',False),('ot',True)]:
            A=gaussian_mapping(pcov,p['cov'],opt)
            out[name]=np.clip(self.pre.inverse(p['mu']+raw@A.T),0,None)
        out['ot_rank']=rank_calibrate_to_style(out['ot'],xs)
        out['ot_constrained']=out['ot_rank'].copy()
        for col,cap in self.caps(style,'attachment_mg_L').items():
            j=self.features.index(col);out['ot_constrained'][:,j]=np.minimum(out['ot_constrained'][:,j],cap)
        return out
    def transfer(self,X,source,target,optimal=True):
        a,b=self.params[source],self.params[target]
        z=self.pre.transform(X);A=gaussian_mapping(a['cov'],b['cov'],optimal)
        return np.clip(self.pre.inverse(b['mu']+(z-a['mu'])@A.T),0,None)


def rank_calibrate_to_style(x,target):
    n,d=x.shape;out=np.empty_like(x)
    for j in range(d):
        q=(rankdata(x[:,j],method='average')-.5)/n
        out[:,j]=np.quantile(target[:,j],q)
    return out


def style_holdout(styles,seed,fraction=.2):
    rng=np.random.default_rng(seed);test=[]
    for g in sorted(set(styles)):
        ix=np.flatnonzero(styles==g);count=max(1,min(len(ix)-2,int(np.ceil(len(ix)*fraction))))
        test.extend(rng.permutation(ix)[:count])
    te=np.sort(test);tr=np.setdiff1d(np.arange(len(styles)),te)
    assert not np.intersect1d(tr,te).size
    assert all(np.sum(styles[tr]==g)>=2 for g in set(styles))
    return tr,te


def tune_sensory(X,Y,seed):
    splits=list(KFold(3,shuffle=True,random_state=seed).split(X))
    data=[]
    for tr,va in splits:
        pre=ConcentrationTransform().fit(X[tr]);data.append((tr,va,pre.transform(X[tr]),pre.transform(X[va])))
    best=(-np.inf,None)
    for name,values in [('pls',[2,4,8,12]),('ridge',[1.,10.,100.,1000.])]:
        for value in values:
            pred=np.zeros_like(Y)
            for tr,va,ztr,zva in data:
                m=PLSRegression(n_components=int(value),scale=False) if name=='pls' else Ridge(alpha=value)
                m.fit(ztr,Y[tr]);pred[va]=m.predict(zva)
            score=float(r2_score(Y,pred))
            if score>best[0]:best=(score,(name,value))
    pre=ConcentrationTransform().fit(X);name,value=best[1]
    m=PLSRegression(n_components=int(value),scale=False) if name=='pls' else Ridge(alpha=value)
    m.fit(pre.transform(X),Y)
    return {'pre':pre,'model':m,'name':name,'value':value,'inner_r2':best[0]}


def predict_sensory(model,X):return model['model'].predict(model['pre'].transform(X))


def mmd2(x,y,gamma):
    kxx=np.exp(-gamma*cdist(x,x,'sqeuclidean'));kyy=np.exp(-gamma*cdist(y,y,'sqeuclidean'))
    cross=np.exp(-gamma*cdist(x,y,'sqeuclidean'));n,m=len(x),len(y)
    return float((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*cross.mean())


def fidelity(a,b):
    return {'sensory_pearson':float(pearsonr(a,b).statistic),
            'sensory_rmse':float(np.sqrt(np.mean((a-b)**2)))}


def defect_metrics(x,features,scenario,source=None):
    rows={}
    for name,(col,tau) in DEFECTS.items():
        tau=UNIT_SCENARIOS[scenario] if name=='T2N' else tau;j=features.index(col);v=x[:,j]
        rows[name+'_above_strict']=float(np.mean(v>tau+1e-12))
        rows[name+'_mean']=float(v.mean());rows[name+'_threshold_numeric']=tau
        if source is not None:
            s=source[:,j]
            rows[name+'_concentration_worse']=float(np.mean(v>s+1e-12))
            rows[name+'_excess_worse']=float(np.mean(np.maximum(v-tau,0)>np.maximum(s-tau,0)+1e-12))
    return rows


def project(raw,source,features,scenario,mode,style_caps):
    x=raw.copy()
    for name,(col,tau) in DEFECTS.items():
        tau=UNIT_SCENARIOS[scenario] if name=='T2N' else tau;j=features.index(col)
        if mode=='strict':cap=np.minimum(source[:,j],tau)
        elif mode=='excess':cap=np.maximum(source[:,j],tau)
        elif mode=='style':cap=style_caps[col]
        else:continue
        x[:,j]=np.minimum(x[:,j],cap)
    return x


def evaluate(generated,generator,style,real,real_y,surrogate,classifier):
    trainz=generator.z[generator.styles==style];rz=generator.pre.transform(real)
    realpairs=pdist(trainz,'sqeuclidean');gamma=1/max(float(np.median(realpairs)),1e-8)
    drr=cdist(trainz,trainz);np.fill_diagonal(drr,np.inf);radius=float(np.median(drr.min(axis=1)))
    inv=np.linalg.inv(generator.params[style]['cov']);mu=generator.params[style]['mu'];rows=[]
    for name,x in generated.items():
        z=generator.pre.transform(x);d=cdist(rz,z);pred=predict_sensory(surrogate,x)
        row={'model':name,'style':style,'n_real':len(real),'n_generated':len(x),
             'mmd2':mmd2(rz,z,gamma),'gamma_train_only':gamma,
             'coverage':float((d.min(axis=1)<=radius).mean()),
             'precision':float((d.min(axis=0)<=radius).mean()),
             'rf_hit':float(np.mean(classifier.predict(z)==style)),
             'mahalanobis_mean':float(np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',z-mu,inv,z-mu),0)).mean()),
             'pair_distance':float(np.median(pdist(z))),
             **fidelity(np.median(pred,axis=0),np.median(real_y,axis=0))}
        for scenario in UNIT_SCENARIOS:
            for key,val in defect_metrics(x,generator.features,scenario).items():row[scenario+'__'+key]=val
        rows.append(row)
    return rows


def _style():
    f=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if f.exists():font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42})


def save(fig,name):
    fig.tight_layout();fig.savefig(IMG_DIR/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(IMG_DIR/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def plots(metrics,generated,portraits,transfer,features,styles,raw):
    folds=metrics.groupby(['repeat','model'])[['mmd2','rf_hit']].mean()
    avg=folds.groupby('model').mean().loc[MODELS];sd=folds.groupby('model').std().loc[MODELS]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    avg.mmd2.plot.bar(ax=axs[0],yerr=sd.mmd2,capsize=3)
    avg.rf_hit.plot.bar(ax=axs[1],yerr=sd.rf_hit,capsize=3)
    axs[0].set_title('独立留出 MMD²（U 统计量可为负）');axs[1].set_title('分类器命中率：辅助诊断')
    save(fig,'q3_mmd_classifier')
    cols=['DMS','X2.3.butanedione','X.E..2.nonenal','X4VG']
    fig,axs=plt.subplots(3,4,figsize=(12,8))
    for i,st in enumerate(TARGET_STYLES):
        for j,col in enumerate(cols):
            ax=axs[i,j];xs=raw[styles==st,features.index(col)];xg=generated[(st,'copula')][:,features.index(col)]
            ax.hist(np.log1p(xs),bins=12,alpha=.5,density=True,label='真实');ax.hist(np.log1p(xg),bins=12,alpha=.5,density=True,label='全量拟合生成')
            ax.set_title(st+' / '+col,fontsize=8)
    axs[0,0].legend(fontsize=7);save(fig,'q3_distributions')
    dims=['A_malt_all','A_hops_all','A_esters_all','acidity','sweetness','alcohol','body','bitternes']
    theta=np.linspace(0,2*np.pi,len(dims),endpoint=False);angles=np.r_[theta,theta[0]]
    fig,axs=plt.subplots(1,3,figsize=(12,4),subplot_kw={'projection':'polar'})
    for ax,st in zip(axs,TARGET_STYLES):
        for kind,model,label in [('real','observed','实测中位数'),('generated','copula','生成代理中位数')]:
            a=portraits[(portraits['style']==st)&(portraits.kind==kind)&(portraits.model==model)][dims].median().values
            # Same fixed affine display shift on both series; values are standardized responses, not grades.
            ax.plot(angles,np.r_[a,a[0]]+3,label=label)
        ax.set_xticks(theta,[TARGET_CN.get(t,t) for t in dims],fontsize=7);ax.set_title(st)
    axs[0].legend(fontsize=7,loc='upper right');save(fig,'q3_radar')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,scenario in zip(axs,UNIT_SCENARIOS):
        a=transfer[transfer.unit_scenario==scenario].groupby('variant')[[n+'_excess_worse' for n in DEFECTS]].mean()
        a.plot.bar(ax=ax);ax.set_title(scenario+'：超阈值程度恶化率');ax.legend(fontsize=6)
    save(fig,'q3_transfer_defects')
    fig,ax=plt.subplots(figsize=(8,4))
    key=['ethanol..v.v.','ethyl_acetate','isopentyl_acetate','linalool']
    profiles=pd.read_csv(OUT_DIR/'transfer_profiles.csv')
    profiles=profiles[profiles.unit_scenario=='attachment_mg_L']
    target=raw[styles==TARGET_STYLE][:,[features.index(c) for c in key]].mean(axis=0)
    axis=np.arange(len(key));width=.18
    for k,variant in enumerate(['source','raw','excess','strict']):
        mean=profiles.loc[profiles.variant==variant,key].mean().values
        ax.bar(axis+(k-1.5)*width,mean/np.maximum(target,1e-12),width,label=variant)
    ax.axhline(1,color='black',ls='--',lw=.8,label='真实 Tripel 均值')
    ax.set_xticks(axis,key);ax.set_ylabel('相对真实 Tripel 均值的比例');ax.legend(fontsize=7)
    ax.set_title('迁移前后关键化合物：分母使用目标风格均值')
    save(fig,'q3_transfer_compounds')
    final=metrics.groupby('model')[[s+'__'+n+'_above_strict' for s in UNIT_SCENARIOS for n in DEFECTS]].mean()
    fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(final.values,aspect='auto',vmin=0,vmax=1)
    fig.colorbar(im,ax=ax,label='超阈值比例')
    ax.set_yticks(range(len(final)),final.index);ax.set_xticks(range(len(final.columns)),final.columns,rotation=80,fontsize=6)
    save(fig,'q3_generation_defects')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--n-gen',type=int,default=300);args=parser.parse_args()
    _style();OUT_DIR.mkdir(exist_ok=True);IMG_DIR.mkdir(exist_ok=True);D=load_all()
    X=D['X_cal'].values;Y=D['y'].values;styles=np.asarray(D['styles']);features=D['cal_features'];beer=D['meta'].beer.values
    metrics=[];split_rows=[];sensory_rows=[];class_rows=[];t0=time.time()
    sensory_oof=[];truth_oof=[]
    for repeat in range(args.repeats):
        tr,te=style_holdout(styles,SEED+repeat)
        for scope,indices in [('train',tr),('test',te)]:
            split_rows.extend(dict(repeat=repeat,scope=scope,beer=int(beer[i]),style=styles[i]) for i in indices)
        gen=FlavorGenerator().fit(X[tr],styles[tr],features)
        surrogate=tune_sensory(X[tr],Y[tr],SEED+repeat)
        sp=predict_sensory(surrogate,X[te]);sensory_oof.append(sp);truth_oof.append(Y[te])
        for j,t in enumerate(D['targets']):
            sensory_rows.append(dict(repeat=repeat,target=t,r2=r2_score(Y[te,j],sp[:,j]),
                model=surrogate['name'],parameter=surrogate['value']))
        classifier=RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',n_jobs=4,random_state=SEED+repeat)
        classifier.fit(gen.z,styles[tr]);pred=classifier.predict(gen.pre.transform(X[te]))
        class_rows.append(dict(repeat=repeat,n_test=len(te),accuracy=float(np.mean(pred==styles[te])),
            balanced_accuracy=balanced_accuracy_score(styles[te],pred)))
        for si,style in enumerate(TARGET_STYLES):
            generated=gen.generate(style,args.n_gen,SEED+repeat*1000+si)
            # n0 sensitivity is a comparison, not a claim of tuned optimality.
            for n0 in [5,50]:
                alt=FlavorGenerator(n0).fit(X[tr],styles[tr],features)
                generated['copula_n0_'+str(n0)]=alt.generate(style,args.n_gen,SEED+repeat*1000+si)['copula']
            mask=styles[te]==style
            for row in evaluate(generated,gen,style,X[te][mask],Y[te][mask],surrogate,classifier):
                row['repeat']=repeat;row['scope']='heldout_real_reference_all_parameters_train_only';metrics.append(row)
        pd.DataFrame(metrics).to_csv(OUT_DIR/'metrics.csv',index=False)
        pd.DataFrame(split_rows).to_csv(OUT_DIR/'splits.csv',index=False)
        print(f'[Q3] holdout {repeat+1}/{args.repeats}; train={len(tr)}, test={len(te)}, sensory={surrogate["name"]}, elapsed={time.time()-t0:.1f}s',flush=True)
    metric_df=pd.DataFrame(metrics);sens=pd.DataFrame(sensory_rows);sens.to_csv(OUT_DIR/'sensory_cv.csv',index=False)
    classes=pd.DataFrame(class_rows);classes.to_csv(OUT_DIR/'classifier_cv.csv',index=False)
    # Refit final descriptive spectra after evaluation; none are scored as independent validation.
    final=FlavorGenerator().fit(X,styles,features);surrogate=tune_sensory(X,Y,SEED)
    generated={};gen_rows=[];portrait_rows=[]
    for i,st in enumerate(styles):
        portrait_rows.append(dict(kind='real',model='observed',style=st,sample_id=int(beer[i]),**dict(zip(D['targets'],Y[i]))))
    for si,st in enumerate(TARGET_STYLES):
        for model,x in final.generate(st,args.n_gen,SEED+si).items():
            generated[(st,model)]=x;py=predict_sensory(surrogate,x)
            for i in range(len(x)):
                ident=f'{st}_{model}_{i:04d}'
                gen_rows.append(dict(style=st,sample_id=ident,generator=model,scope='all_data_fit_descriptive',**dict(zip(features,x[i]))))
                portrait_rows.append(dict(kind='generated',model=model,style=st,sample_id=ident,**dict(zip(D['targets'],py[i]))))
    src=final.pre.clean(X[styles==SOURCE_STYLE]);ids=beer[styles==SOURCE_STYLE]
    raw=final.transfer(src,SOURCE_STYLE,TARGET_STYLE);wc=final.transfer(src,SOURCE_STYLE,TARGET_STYLE,False)
    tr_rows=[];tr_metrics=[];stale_col=D['targets'].index('stale_hops');src_pred=predict_sensory(surrogate,src)
    for scenario in UNIT_SCENARIOS:
        caps=final.caps(TARGET_STYLE,scenario)
        variants={'source':src,'covmatch':wc,'raw':raw}
        for mode in ['strict','excess','style']:variants[mode]=project(raw,src,features,scenario,mode,caps)
        for variant,x in variants.items():
            pred=predict_sensory(surrogate,x)
            row=dict(unit_scenario=scenario,variant=variant,n_source=len(src),
                **defect_metrics(x,features,scenario,source=src),
                **fidelity(np.median(pred,axis=0),np.median(Y[styles==TARGET_STYLE],axis=0)),
                stale_hops_predicted_change_mean=float((pred[:,stale_col]-src_pred[:,stale_col]).mean()),
                stale_hops_guarantee=False,scope='all_data_fit_descriptive_transfer')
            tr_metrics.append(row)
            for i in range(len(x)):
                tr_rows.append(dict(unit_scenario=scenario,variant=variant,style=SOURCE_STYLE if variant=='source' else TARGET_STYLE,
                    sample_id=int(ids[i]),**dict(zip(features,x[i]))))
                portrait_rows.append(dict(kind='transfer',model=scenario+'_'+variant,style=TARGET_STYLE,
                    sample_id=int(ids[i]),**dict(zip(D['targets'],pred[i]))))
            if variant=='strict':
                assert all(row[n+'_concentration_worse']==0 and row[n+'_above_strict']==0 for n in DEFECTS)
            if variant=='excess':assert all(row[n+'_excess_worse']==0 for n in DEFECTS)
    trm=pd.DataFrame(tr_metrics);ports=pd.DataFrame(portrait_rows)
    pd.DataFrame(gen_rows).to_csv(OUT_DIR/'generated_profiles.csv',index=False,float_format='%.10g')
    pd.DataFrame(tr_rows).to_csv(OUT_DIR/'transfer_profiles.csv',index=False,float_format='%.10g')
    trm.to_csv(OUT_DIR/'transfer_metrics.csv',index=False);ports.to_csv(OUT_DIR/'portraits.csv',index=False,float_format='%.10g')
    averaged=metric_df.groupby(['model','style']).mean(numeric_only=True)
    repeat_means=metric_df.groupby(['repeat','model'])[['mmd2','coverage','precision','rf_hit','sensory_rmse']].mean()
    repeat_means.to_csv(OUT_DIR/'model_comparison_folds.csv')
    agg=repeat_means.groupby('model').agg(['mean','std'])
    agg.to_csv(OUT_DIR/'model_comparison.csv');print(agg.to_string(),flush=True)
    summary={'schema_version':2,'provenance':provenance(),'runtime_sec':time.time()-t0,
        'meta':{'n_features':len(features),'n_samples':len(X),'holdout_repeats':args.repeats,'n_gen':args.n_gen,
            'transform':'training-median impute; log1p(c/training_positive_median); training standardization',
            'holdout':'per style ceil(0.2*n), at least 1 test and 2 train; repeated splits overlap; SD is not an independent-sample CI',
            'primary_generator':'copula predeclared; comparisons do not establish global optimality',
            'scope':'metrics.csv uses heldout real references; final spectra and transfer are descriptive all-data refits',
            't2n_units':'UNRESOLVED; primary retains attachment mg/L with tau=0.0001; alternative assumes ug/L with tau=0.1'},
        'generation':{'models':MODELS,'metrics':{m+'|'+st:{k:float(v) for k,v in r.items()} for (m,st),r in averaged.iterrows()}},
        'sensory_model':{'outer_test_r2_macro_mean':float(sens.r2.mean()),
            'pooled_repeated_holdout_r2_macro':float(r2_score(np.vstack(truth_oof),np.vstack(sensory_oof))),
            'stale_hops_r2':float(sens.loc[sens.target=='stale_hops','r2'].mean()),
            'warning':'Surrogate predictions are not observed grades; no stale-hops guarantee'},
        'classifier':classes.to_dict('records'),'transfer':{'metrics':trm.to_dict('records'),
            'primary_constraint':'excess nonincrease; strict is stronger concentration nonincrease plus below threshold'},
        'limitations':['Absolute sensory grades unidentified','T2N unit unresolved',
            'No valid chemical proxy guarantee for stale hops','No physical brewing feasibility guarantee',
            'Small heldout style sizes; MMD2 U statistic may be negative','Given taste-panel data do not identify a causal paired transfer']}
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    plots(metric_df,generated,ports,trm,features,styles,final.x)
    print(f'[Q3] complete in {time.time()-t0:.1f}s',flush=True)


if __name__=='__main__':main()
