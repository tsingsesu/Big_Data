from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import 公共数据处理 as du
ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "data/external"
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
    import io,zipfile,urllib.request
    EXT.mkdir(exist_ok=True)
    path=EXT/'外部啤酒数据.csv'
    if not path.exists():
        url='https://www.kaggle.com/api/v1/datasets/download/stephenpolozoff/top-beer-information'
        with urllib.request.urlopen(url,timeout=60) as response:blob=response.read(20_000_000)
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            names = {'beer_data_set.csv': '外部啤酒数据.csv',
                     'Beer Descriptors Simplified.xlsx': '外部感官描述说明.xlsx'}
            for member in z.namelist():
                name = names.get(Path(member).name)
                if name:
                    (EXT/name).write_bytes(z.read(member))
    raw=pd.read_csv(path);cols=['Astringency','Body','Alcohol','Bitter','Sweet','Sour','Salty','Fruits','Hoppy','Spices','Malty']
    mapping=raw[['Style']].drop_duplicates().sort_values('Style');mapping['attachment_style']=mapping.Style.map(map_style)
    mapping.to_csv(EXT/'外部风格映射.csv',index=False)
    raw['style']=raw.Style.map(map_style);total=raw[cols].sum(axis=1)
    keep=total.ge(50)&raw['style'].notna();fraction=raw.loc[keep,cols].div(total[keep],axis=0)
    fraction['风格']=raw.loc[keep,'style'];p=fraction.groupby('风格')[cols].mean()
    p.insert(0,'n_kept',fraction.groupby('风格').size());p=p.rename(columns={'Astringency':'Astr','Alcohol':'Alc'}).reset_index()
    p.to_csv(EXT/'外部风格画像.csv',index=False,float_format='%.12g')
    assert np.allclose(p.drop(columns=['风格','n_kept']).sum(axis=1),1.)
    print('[external] 原始行数、映射行数、筛选行数、风格数：', len(raw), int(raw['style'].notna().sum()), int(keep.sum()), len(p), flush=True)
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
    val.to_csv(EXT / "外部风格验证.csv", index=False)
    print(val.round(3).to_string(index=False))

    print("Saved external style mapping, prototypes and validation tables")


if __name__ == "__main__":
    main()
