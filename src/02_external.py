"""外部数据校验：Kaggle 风格级感官原型 vs 附件 4.csv / 7.csv 的风格均值。

运行：
    MPLBACKEND=Agg PYTHONPATH=.pylibs:src python3 src/02_external.py
输出：
    data/external/style_validation.csv
    img/external_style_validation.png/.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import data_utils as du

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
EXT = ROOT / "data" / "external"
IMG.mkdir(exist_ok=True)

from matplotlib import font_manager
_cjk_font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if _cjk_font.exists():
    font_manager.fontManager.addfont(str(_cjk_font))
plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False, 'pdf.fonttype': 42})

plt.rcParams.update({"font.size": 8, "savefig.dpi": 300, "figure.dpi": 150})


def safe_r(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan
    if method == "pearson":
        r, p = pearsonr(x[m], y[m])
    else:
        r, p = spearmanr(x[m], y[m])
    return float(r), float(p)


def map_style(style):
    if style=='Lager - India Pale Lager (IPL)':return 'Hoppy'
    if style=='Lager - American Amber / Red':return 'Amber'
    if style.startswith(('Lager -','Pilsner -')):return 'Pils/Lager'
    if style.startswith(('Stout -','Porter -')):return 'Stout/Porter'
    if style.startswith('Wheat Beer -'):return 'Wheat'
    if style.startswith('IPA -') or style in ['Pale Ale - American','Pale Ale - English','Pale Ale - Belgian']:return 'Hoppy'
    exact={
      'Blond':['Blonde Ale - American','Blonde Ale - Belgian','Kölsch'],
      'Amber':['Altbier','Red Ale - American Amber / Red','Red Ale - Irish'],
      'Brown':['Brown Ale - American','Brown Ale - Belgian Dark','Brown Ale - English','Mild Ale - English Dark','Mild Ale - English Pale','Old Ale'],
      'Strong ale':['Strong Ale - American','Strong Ale - English','Strong Ale - Belgian Dark','Strong Ale - Belgian Pale','Quadrupel (Quad)','Barleywine - American','Barleywine - English','Red Ale - Imperial','Bock - Doppelbock','Bock - Eisbock'],
      'Tripel':['Tripel'],'Dubbel':['Dubbel'],'Lambic':['Lambic - Traditional','Lambic - Gueuze'],
      'Kriek':['Lambic - Fruit'],'Faro':['Lambic - Faro'],
      'Saison':['Farmhouse Ale - Saison','Farmhouse Ale - Bière de Garde','Farmhouse Ale - Sahti'],
      'Christmas':['Winter Warmer','Herb and Spice Beer','Pumpkin Beer','Chile Beer'],
      'Low/No alcohol':['Low Alcohol Beer'],'Fruitbeer':['Fruit and Field Beer'],
      'West Flanders ale':['Sour - Flanders Red Ale'],'Flanders old brown':['Sour - Flanders Oud Bruin'],
      'Brett/cofermented':['Brett Beer','Wild Ale'],'Scotch':['Scotch Ale / Wee Heavy','Scottish Ale'],
      'Brut':['Bière de Champagne / Bière Brut']}
    return next((k for k,v in exact.items() if style in v),None)


def build_prototypes():
    import json,hashlib,io,zipfile,urllib.request
    EXT.mkdir(exist_ok=True)
    path=EXT/'beer_data_set.csv'
    if not path.exists():
        url='https://www.kaggle.com/api/v1/datasets/download/stephenpolozoff/top-beer-information'
        with urllib.request.urlopen(url,timeout=60) as response:blob=response.read(20_000_000)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for member in z.namelist():
                if Path(member).name in ['beer_data_set.csv','Beer Descriptors Simplified.xlsx']:
                    (EXT/Path(member).name).write_bytes(z.read(member))
    raw=pd.read_csv(path);cols=['Astringency','Body','Alcohol','Bitter','Sweet','Sour','Salty','Fruits','Hoppy','Spices','Malty']
    mapping=raw[['Style']].drop_duplicates().sort_values('Style');mapping['attachment_style']=mapping.Style.map(map_style)
    mapping.to_csv(EXT/'style_mapping.csv',index=False)
    raw['style']=raw.Style.map(map_style);total=raw[cols].sum(axis=1)
    keep=total.ge(50)&raw['style'].notna();fraction=raw.loc[keep,cols].div(total[keep],axis=0)
    fraction['风格']=raw.loc[keep,'style'];p=fraction.groupby('风格')[cols].mean()
    p.insert(0,'n_kept',fraction.groupby('风格').size());p=p.rename(columns={'Astringency':'Astr','Alcohol':'Alc'}).reset_index()
    p.to_csv(EXT/'style_prototypes.csv',index=False,float_format='%.12g')
    assert np.allclose(p.drop(columns=['风格','n_kept']).sum(axis=1),1.)
    meta={'n_raw':len(raw),'n_raw_styles':raw.Style.nunique(),'n_mapped_styles':mapping.attachment_style.notna().sum(),
        'n_mapped_rows':int(raw['style'].notna().sum()),'n_quality_rows_all':int(total.ge(50).sum()),
        'n_mapped_quality_rows':int(keep.sum()),'n_target_styles':len(p),
        'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source':'https://www.kaggle.com/datasets/stephenpolozoff/top-beer-information',
        'license_recorded':'CC BY 4.0; project 2026-10-24 API retrieval record, not re-verified online',
        'role':'style-level external validation only; not Q1 or Q3 training input',
        'mapping':'hand-authored mutually exclusive semantic mapping; source table exported',
        'normalization':'11 word counts / row total; total>=50'}
    meta={k:int(v) if isinstance(v,np.integer) else v for k,v in meta.items()}
    (EXT/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('[external]',json.dumps(meta,ensure_ascii=False),flush=True)
    return p


def main() -> None:
    D = du.load_all()
    df4, df7 = D["df4"], D["df7"]
    proto = build_prototypes().rename(columns={"风格": "style"})
    styles = proto["style"].tolist()

    to4 = [
        "acidity", "bitternes", "sweetness", "alcohol",
        "A_malt_all", "A_hops_all", "A_esters_fruity", "clove", "coriander", "body",
    ]
    to7 = [
        "taste_sour", "taste_bitter", "taste_sweet", "taste_alcohol",
        "aroma_malt", "aroma_hoppy", "aroma_fruity", "aroma_clove", "aroma_coriander", "palate_blanket",
    ]
    m4 = df4.groupby("tasting_category_fine")[to4].mean().reindex(styles)
    m7 = df7.groupby("tasting_category_fine")[to7].mean().reindex(styles)

    # Spices 用 clove + coriander 的均值
    m4["spices"] = m4[["clove", "coriander"]].mean(axis=1)
    m7["spices"] = m7[["aroma_clove", "aroma_coriander"]].mean(axis=1)

    pairs = [
        ("Sour", "Sour", "acidity", "taste_sour"),
        ("Bitter", "Bitter", "bitternes", "taste_bitter"),
        ("Sweet", "Sweet", "sweetness", "taste_sweet"),
        ("Alcohol", "Alc", "alcohol", "taste_alcohol"),
        ("Malty", "Malty", "A_malt_all", "aroma_malt"),
        ("Hoppy", "Hoppy", "A_hops_all", "aroma_hoppy"),
        ("Fruits", "Fruits", "A_esters_fruity", "aroma_fruity"),
        ("Spices", "Spices", "spices", "spices"),
        ("Body", "Body", "body", "palate_blanket"),
    ]

    rows = []
    for name, kcol, c4, c7 in pairs:
        x = proto[kcol].values
        r4, p4 = safe_r(x, m4[c4].values)
        r7, p7 = safe_r(x, m7[c7].values)
        s4, _ = safe_r(x, m4[c4].values, "spearman")
        s7, _ = safe_r(x, m7[c7].values, "spearman")
        # leave-one-style-out 范围（Pearson）
        loo4, loo7 = [], []
        for i in range(len(styles)):
            mask = np.arange(len(styles)) != i
            rr4, _ = safe_r(x[mask], m4[c4].values[mask])
            rr7, _ = safe_r(x[mask], m7[c7].values[mask])
            if np.isfinite(rr4):
                loo4.append(rr4)
            if np.isfinite(rr7):
                loo7.append(rr7)
        rows.append({
            "dimension": name,
            "kaggle_col": kcol,
            "target_4csv": c4,
            "ratebeer_7csv": c7,
            "pearson_4csv": r4,
            "spearman_4csv": s4,
            "pearson_7csv": r7,
            "spearman_7csv": s7,
            "loo4_min": np.min(loo4) if loo4 else np.nan,
            "loo4_max": np.max(loo4) if loo4 else np.nan,
            "loo7_min": np.min(loo7) if loo7 else np.nan,
            "loo7_max": np.max(loo7) if loo7 else np.nan,
            "n_styles": len(styles),
        })
    val = pd.DataFrame(rows)
    val.to_csv(EXT / "style_validation.csv", index=False)
    print(val.round(3).to_string(index=False))

    def plot_scatter(m, filename, ylabel, title):
        fig, axes = plt.subplots(3, 3, figsize=(8.2, 7.0))
        for ax, (name, kcol, c4, c7) in zip(axes.ravel(), pairs):
            x = proto[kcol].values
            y = m[c4 if m is m4 else c7].values
            ax.scatter(x, y, s=22, color="#4C72B0", alpha=0.85)
            for s, xi, yi in zip(styles, x, y):
                if s in ("Tripel", "Kriek", "Lambic", "Pils/Lager"):
                    ax.annotate(s, (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
            r, _ = safe_r(x, y)
            ax.set_title(f"{name}: r={r:.2f}", fontsize=8)
            ax.set_xlabel("Kaggle prototype (word share)", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(title, y=1.01, fontsize=10)
        fig.tight_layout()
        fig.savefig(IMG / f"{filename}.png", bbox_inches="tight")
        fig.savefig(IMG / f"{filename}.pdf", bbox_inches="tight")
        plt.close(fig)

    plot_scatter(
        m4, "external_style_validation",
        "4.csv style mean",
        "External validity: Kaggle prototypes vs trained panel (4.csv), 22 styles",
    )
    plot_scatter(
        m7, "external_style_validation_7csv",
        "7.csv style mean",
        "External validity: Kaggle prototypes vs Ratebeer descriptors (7.csv), 22 styles",
    )
    print("\nSaved data/external/style_validation.csv and img/external_style_validation*.*")


if __name__ == "__main__":
    main()
