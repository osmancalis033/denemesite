"""
BIST & ABD Hisse Analiz Paneli
Gereksinimler: pip install streamlit yfinance plotly pandas numpy requests
Çalıştırma  : streamlit run borsa_analiz.py
""" 

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Sayfa Ayarları ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📈 Hisse Analiz Paneli",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-box {
        background: #1a1f2e; border: 1px solid #2d3748;
        border-radius: 12px; padding: 14px 16px;
        text-align: center; margin-bottom: 8px;
    }
    .metric-label { font-size: 10px; color: #8892a4; text-transform: uppercase;
        letter-spacing: 0.8px; margin-bottom: 5px; }
    .metric-value { font-size: 18px; font-weight: 700; color: #e2e8f0; }
    .metric-value.positive { color: #48bb78; }
    .metric-value.negative { color: #fc8181; }
    .section-title { font-size: 15px; font-weight: 600; color: #a0aec0;
        border-left: 3px solid #4299e1; padding-left: 10px; margin: 20px 0 12px 0; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600; margin: 3px; }
    .badge-buy  { background: #276749; color: #9ae6b4; }
    .badge-sell { background: #742a2a; color: #feb2b2; }
    .badge-hold { background: #744210; color: #fbd38d; }
    .stock-header { background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2d3748; border-radius: 16px;
        padding: 22px 26px; margin-bottom: 18px; }
    .data-table { width: 100%; border-collapse: collapse; }
    .data-table td { padding: 8px 12px; border-bottom: 1px solid #1e2435; font-size: 13px; }
    .data-table td:first-child { color: #8892a4; }
    .data-table td:last-child { color: #e2e8f0; font-weight: 500; text-align: right; }
    .info-box { background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 14px 18px; margin-bottom: 14px; font-size: 13px; color: #8892a4; }
    .tag { display: inline-block; background: #2d3748; color: #90cdf4;
        border-radius: 6px; padding: 2px 8px; font-size: 11px; margin: 2px; }
    .stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px;
        gap: 4px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8892a4; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #2d3748 !important; color: #e2e8f0 !important; }
    div[data-testid="stDataFrame"] table { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def fmt_num(n, dec=2, suffix=""):
    if n is None: return "—"
    try:
        n = float(n)
        if np.isnan(n): return "—"
        if abs(n) >= 1e12: return f"{n/1e12:.{dec}f}T{suffix}"
        if abs(n) >= 1e9:  return f"{n/1e9:.{dec}f}B{suffix}"
        if abs(n) >= 1e6:  return f"{n/1e6:.{dec}f}M{suffix}"
        return f"{n:,.{dec}f}{suffix}"
    except Exception: return str(n)

def safe(d, *keys, default="—"):
    for k in keys:
        try:
            v = d.get(k)
            if v is None or v == "": continue
            if isinstance(v, float) and np.isnan(v): continue
            return v
        except Exception: pass
    return default

def pct(v, x100=True):
    if v == "—" or v is None: return "—"
    try:
        f = float(v)
        if np.isnan(f): return "—"
        return f"{f*100:.2f}%" if x100 else f"{f:.2f}%"
    except Exception: return str(v)

def fv(v, fmt=".1f", suf=""):
    if v == "—": return "—"
    try: return f"{float(v):{fmt}}{suf}"
    except Exception: return str(v)

def mcard(label, value, cls="", sub=""):
    s = f'<div style="font-size:11px;color:#718096;margin-top:3px">{sub}</div>' if sub else ""
    return f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value {cls}">{value}</div>{s}</div>'

def last_val(s):
    v = s.dropna()
    return float(v.iloc[-1]) if not v.empty else None

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def calc_macd(s, f=12, sl=26, sg=9):
    m = s.ewm(span=f,adjust=False).mean() - s.ewm(span=sl,adjust=False).mean()
    sig = m.ewm(span=sg,adjust=False).mean()
    return m, sig, m - sig

def calc_bb(s, n=20, k=2):
    sma = s.rolling(n).mean()
    std = s.rolling(n).std()
    return sma+k*std, sma, sma-k*std


# ─── Veri Fonksiyonları ───────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_info_and_hist(sym: str, period: str, is_us: bool = False):
    yf_sym = sym if (is_us or sym.endswith(".IS")) else sym + ".IS"
    stk  = yf.Ticker(yf_sym)
    info = dict(stk.info) if stk.info else {}
    hist = stk.history(period=period)
    return info, hist

def get_ticker(sym: str, is_us: bool = False):
    yf_sym = sym if (is_us or sym.endswith(".IS")) else sym + ".IS"
    return yf.Ticker(yf_sym)

@st.cache_data(ttl=300)
def get_major_holders(sym: str, is_us: bool = False):
    yf_sym = sym if (is_us or sym.endswith(".IS")) else sym + ".IS"
    try:
        mh = yf.Ticker(yf_sym).major_holders
        if mh is None or mh.empty: return []
        result = []
        for i in range(len(mh)):
            try:
                row = mh.iloc[i]
                val = row.iloc[0] if len(row) > 0 else None
                lbl = row.iloc[1] if len(row) > 1 else f"Kalem {i}"
                if val is not None:
                    result.append((str(lbl), str(val)))
            except Exception: continue
        return result
    except Exception: return []

@st.cache_data(ttl=300)
def get_inst_holders(sym: str, is_us: bool = False):
    yf_sym = sym if (is_us or sym.endswith(".IS")) else sym + ".IS"
    try:
        ih = yf.Ticker(yf_sym).institutional_holders
        return None if (ih is None or ih.empty) else ih
    except Exception: return None

@st.cache_data(ttl=300)
def get_dividends(sym: str, is_us: bool = False):
    yf_sym = sym if (is_us or sym.endswith(".IS")) else sym + ".IS"
    try:
        d = yf.Ticker(yf_sym).dividends
        return None if (d is None or d.empty) else d
    except Exception: return None

@st.cache_data(ttl=300)
def get_analyst_rec(sym: str, is_us: bool = False):
    yf_sym = sym if (is_us or sym.endswith(".IS")) else sym + ".IS"
    try:
        r = yf.Ticker(yf_sym).recommendations
        return None if (r is None or r.empty) else r
    except Exception: return None


# ─── ABD Hisse Screener ───────────────────────────────────────────────────────

# Hazır hisse listeleri (yfinance screener API her ortamda çalışmaz,
# güvenilir fallback listeler)
US_PRESETS = {
    "🔵 Large Cap (S&P 500 Seçkisi)": [
        "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","BRK-B","JPM","V",
        "UNH","XOM","JNJ","MA","PG","AVGO","HD","MRK","CVX","ABBV",
        "COST","LLY","PEP","KO","WMT","BAC","ADBE","CRM","TMO","NFLX",
    ],
    "🟠 Mid Cap Seçkisi": [
        "DECK","WSM","LULU","TRMB","RNR","MTDR","BWXT","TXRH","CBSH","EME",
        "KNSL","ALLE","AAON","FIVE","MOH","MANH","NVT","NBIX","SAIA","CASY",
        "TREX","HLNE","PEGA","GMED","LOPE","HRI","AWI","ITCI","SLG","WDFC",
    ],
    "🟢 Small Cap Seçkisi": [
        "SMPL","CSWI","MGNI","PCVX","HIMS","AVAV","CVCO","SKYW","KFY","MYRG",
        "OSIS","STEP","BOOT","DFIN","ATRC","TTGT","PRG","MGRC","CPRX","FORM",
    ],
    "💻 Teknoloji": [
        "AAPL","MSFT","NVDA","GOOGL","META","TSLA","AVGO","ADBE","CRM","ORCL",
        "AMD","INTC","QCOM","TXN","MU","AMAT","KLAC","LRCX","PANW","SNOW",
    ],
    "🏦 Finans & Bankacılık": [
        "JPM","BAC","WFC","GS","MS","C","BLK","AXP","SCHW","CB",
        "PGR","MET","AFL","ALL","TRV","COF","USB","PNC","TFC","FITB",
    ],
    "⚕️ Sağlık": [
        "UNH","JNJ","LLY","MRK","ABBV","TMO","ABT","DHR","BMY","AMGN",
        "GILD","VRTX","REGN","ISRG","SYK","ELV","CI","HUM","CVS","BIIB",
    ],
    "⚡ Enerji": [
        "XOM","CVX","COP","EOG","SLB","PXD","VLO","PSX","MPC","HES",
        "OKE","WMB","ET","KMI","DVN","FANG","APA","HAL","BKR","LNG",
    ],
    "🛍️ Tüketim": [
        "AMZN","WMT","COST","HD","LOW","TGT","TJX","BKNG","MCD","SBUX",
        "NKE","DG","DLTR","BBY","ROST","CMG","YUM","EL","CL","PG",
    ],
    "📡 İletişim & Medya": [
        "META","GOOGL","NFLX","DIS","CMCSA","T","VZ","CHTR","TMUS","EA",
        "TTWO","WBD","FOXA","PARA","LUMN","SIRI","IPG","OMC","NWS","ZNGA",
    ],
    "🏭 Sanayi": [
        "GE","CAT","BA","HON","UPS","LMT","RTX","DE","MMM","EMR",
        "ETN","PH","ITW","GD","NOC","CSX","UNP","NSC","FDX","DAL",
    ],
    "📈 Yüksek Temettü": [
        "T","VZ","MO","PM","ABBV","XOM","CVX","OKE","MPC","VLO",
        "KMI","WMB","ET","EPD","MPLX","IBM","PFE","BMY","WBA","SPOK",
    ],
    "🚀 Büyüme Hisseleri": [
        "NVDA","META","AMZN","GOOGL","TSLA","MSFT","NFLX","AVGO","CRM","SNOW",
        "PLTR","DDOG","NET","CRWD","MDB","ABNB","UBER","LYFT","DASH","RBLX",
    ],
}


BIST_PRESETS = {
    "🏦 Bankacılık": [
        "GARAN","AKBNK","YKBNK","ISCTR","HALKB","VAKBN","QNBFB","ALBRK","SKBNK","KLNMA",
    ],
    "✈️ Havacılık & Ulaşım": [
        "THYAO","PGSUS","MAVI","CLEBI","TAVHL","SASA","BIMAS","ULKER","TTRAK","TCELL",
    ],
    "⚡ Enerji & Elektrik": [
        "ENKAI","AKSEN","ZOREN","EREGL","KRDMD","TUPRS","AYGAZ","BIMAS","DOHOL","SISE",
    ],
    "🏗️ İnşaat & Gayrimenkul": [
        "ENKAI","TEKFEN","TOASO","FROTO","OTKAR","KCHOL","SAHOL","SISE","DOHOL","TTRAK",
    ],
    "💻 Teknoloji & Telecom": [
        "TCELL","TTKOM","LOGO","KAREL","NETAS","INDES","LINK","ARMDA","DGATE","ESCOM",
    ],
    "🛍️ Perakende & Tüketim": [
        "BIMAS","MIGROS","SOKM","ULKER","AEFES","CCOLA","KOZAA","MAVI","DESA","VAKKO",
    ],
    "🏭 Sanayi & Holding": [
        "KCHOL","SAHOL","DOHOL","SISE","TOASO","FROTO","OTKAR","EKGYO","TTRAK","PETKM",
    ],
    "💊 Sağlık & İlaç": [
        "SELEC","MPARK","ACSEL","ECZYT","DEVA","ILAC","TABGD","BIOEN","ADEL","GLYHO",
    ],
    "📈 BIST-30 Seçkisi": [
        "GARAN","AKBNK","THYAO","EREGL","BIMAS","KCHOL","SAHOL","YKBNK","SISE","TCELL",
        "TUPRS","ISCTR","ASELS","FROTO","TOASO","PETKM","KOZAA","PGSUS","HALKB","VAKBN",
        "SASA","ENKAI","TTKOM","MGROS","AEFES","CCOLA","AKSEN","DOHOL","EKGYO","OTKAR",
    ],
    "🏆 Yüksek Temettü (BIST)": [
        "TUPRS","AYGAZ","DOHOL","KCHOL","SAHOL","ENKAI","EREGL","BIMAS","KOZAA","TTKOM",
    ],
}

@st.cache_data(ttl=600)
def get_bist_screener_data(tickers: tuple) -> pd.DataFrame:
    """Verilen BIST ticker listesi için temel metrikleri çek."""
    rows = []
    for t in tickers:
        try:
            info = dict(yf.Ticker(t + ".IS").info)
            rows.append({
                "Sembol":        t,
                "Şirket":        (info.get("shortName") or info.get("longName") or t)[:28],
                "Sektör":        info.get("sector","—") or "—",
                "Fiyat (₺)":     info.get("currentPrice") or info.get("regularMarketPrice"),
                "Değ. %":        info.get("regularMarketChangePercent"),
                "Piyasa Değeri": info.get("marketCap"),
                "F/K":           info.get("trailingPE"),
                "F/DD":          info.get("priceToBook"),
                "ROE %":         info.get("returnOnEquity"),
                "Temettü %":     info.get("dividendYield"),
                "52H Yük (₺)":   info.get("fiftyTwoWeekHigh"),
                "52H Düş (₺)":   info.get("fiftyTwoWeekLow"),
                "Beta":          info.get("beta"),
            })
        except Exception:
            rows.append({"Sembol": t, "Şirket": t})
    return pd.DataFrame(rows)

@st.cache_data(ttl=600)
def get_us_screener_data(tickers: tuple) -> pd.DataFrame:
    """Verilen ticker listesi için temel metrikleri çek."""
    rows = []
    for t in tickers:
        try:
            info = dict(yf.Ticker(t).info)
            rows.append({
                "Sembol":          t,
                "Şirket":          (info.get("shortName") or info.get("longName") or t)[:28],
                "Sektör":          info.get("sector", "—") or "—",
                "Fiyat ($)":       info.get("currentPrice") or info.get("regularMarketPrice"),
                "Değ. %":          info.get("regularMarketChangePercent"),
                "Piyasa Değeri":   info.get("marketCap"),
                "F/K":             info.get("trailingPE"),
                "F/DD":            info.get("priceToBook"),
                "ROE %":           info.get("returnOnEquity"),
                "Temettü %":       info.get("dividendYield"),
                "52H Yük ($)":     info.get("fiftyTwoWeekHigh"),
                "52H Düş ($)":     info.get("fiftyTwoWeekLow"),
                "Beta":            info.get("beta"),
            })
        except Exception:
            rows.append({"Sembol": t, "Şirket": t})
    df = pd.DataFrame(rows)
    return df



# ─── TEFAS Fonksiyonları ──────────────────────────────────────────────────────

# ─── TEFAS: tefas-crawler kütüphanesi ────────────────────────────────────────
# pip install tefas-crawler
# Dökümantasyon: https://pypi.org/project/tefas-crawler/

def _tefas_crawler_available():
    try:
        from tefas import Crawler   # noqa
        return True
    except ImportError:
        return False

@st.cache_data(ttl=1800)
def tefas_fetch_history(code: str, days: int = 90) -> pd.DataFrame:
    """tefas-crawler ile NAV geçmişi + portföy dağılımı sütunlarıyla çek."""
    from tefas import Crawler
    end   = datetime.today()
    start = end - timedelta(days=days)
    tefas = Crawler()
    df = tefas.fetch(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        name=code.upper(),
    )
    return df if df is not None and not df.empty else pd.DataFrame()

@st.cache_data(ttl=1800)
def tefas_fetch_single_day(code: str) -> dict:
    """En son gün verisi (dict)."""
    from tefas import Crawler
    end   = datetime.today()
    start = end - timedelta(days=7)
    tefas = Crawler()
    df = tefas.fetch(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        name=code.upper(),
    )
    if df is not None and not df.empty:
        return df.sort_values("date").iloc[-1].to_dict()
    return {}

@st.cache_data(ttl=3600)
def get_comparison_data(days: int = 90) -> dict:
    """Altın, Dolar, BIST100 ve mevduat faizi geçmiş verisi."""
    end   = datetime.today()
    start = end - timedelta(days=days + 10)
    result = {}
    benchmarks = {
        "BIST-100":  "XU100.IS",
        "Dolar/TL":  "USDTRY=X",
        "Altın/TL":  "GC=F",        # Ons altın — XAUUSD * USDTRY ile çevrilir
        "Gram Altın":"GLDTR.IS",    # Türkiye Altın ETF
    }
    for label, ticker in benchmarks.items():
        try:
            h = yf.Ticker(ticker).history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            if not h.empty:
                result[label] = h["Close"]
        except Exception:
            pass
    return result



# ═══════════════════════════════════════════════════════════════════════════════
# PAYLAŞIMLI BÖLÜMLER: grafik, sinyal, pivot, analist, temettü
# ═══════════════════════════════════════════════════════════════════════════════

def render_technical_tab(sym: str, period: str, is_us: bool = False):
    """Teknik analiz sekmesi — hem BIST hem ABD için ortak."""
    info, hist = get_info_and_hist(sym, period, is_us)
    if hist.empty:
        st.error("Fiyat verisi bulunamadı.")
        return
    close = hist["Close"]
    high  = hist["High"]
    low   = hist["Low"]
    vol   = hist["Volume"]
    cur   = float(close.iloc[-1])

    rsi                   = calc_rsi(close)
    macd_l, sig_l, mhist  = calc_macd(close)
    bb_u, bb_m, bb_l      = calc_bb(close)
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    co1, co2 = st.columns([3, 1])
    with co1:
        inds = st.multiselect("Göstergeler",
            ["Bollinger Bantları","SMA 20","SMA 50","SMA 200"],
            default=["Bollinger Bantları","SMA 20"],
            key=f"inds_{sym}")
    with co2:
        ctype = st.selectbox("", ["Mum","Çizgi"], label_visibility="collapsed", key=f"ct_{sym}")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.14, 0.18, 0.18],
        subplot_titles=("Fiyat","Hacim","RSI (14)","MACD (12,26,9)"),
    )
    # Fiyat
    if ctype == "Mum":
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=high, low=low, close=close,
            increasing_line_color="#48bb78", decreasing_line_color="#fc8181",
            name="Fiyat", showlegend=False), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=hist.index, y=close, mode="lines",
            line=dict(color="#4299e1",width=2), name="Kapanış"), row=1, col=1)

    if "Bollinger Bantları" in inds:
        for y_, nm_, c_ in [(bb_u,"BB Üst","rgba(246,224,94,0.7)"),
                             (bb_m,"BB Orta","rgba(246,224,94,0.35)"),
                             (bb_l,"BB Alt","rgba(246,224,94,0.7)")]:
            fig.add_trace(go.Scatter(x=hist.index, y=y_, mode="lines",
                line=dict(color=c_,width=1,dash="dot"),
                name=nm_, showlegend=(nm_=="BB Üst")), row=1, col=1)

    for ln_, da_, cl_ in [("SMA 20",sma20,"#f6e05e"),("SMA 50",sma50,"#68d391"),("SMA 200",sma200,"#fc8181")]:
        if ln_ in inds:
            fig.add_trace(go.Scatter(x=hist.index, y=da_, mode="lines",
                line=dict(color=cl_,width=1.5), name=ln_), row=1, col=1)

    # Hacim
    vc = ["#48bb78" if c >= o else "#fc8181" for c, o in zip(hist["Close"], hist["Open"])]
    fig.add_trace(go.Bar(x=hist.index, y=vol, marker_color=vc, opacity=0.7,
                         name="Hacim", showlegend=False), row=2, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=hist.index, y=rsi, mode="lines",
        line=dict(color="#9f7aea",width=1.8), name="RSI"), row=3, col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(252,129,129,0.07)",line_width=0,row=3,col=1)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(72,187,120,0.07)", line_width=0,row=3,col=1)
    for y_, c_ in [(70,"rgba(252,129,129,0.45)"),(50,"rgba(100,100,100,0.3)"),(30,"rgba(72,187,120,0.45)")]:
        fig.add_hline(y=y_, line_dash="dot", line_color=c_, line_width=1, row=3, col=1)
    # MACD
    mc = ["#48bb78" if v >= 0 else "#fc8181" for v in mhist.fillna(0)]
    fig.add_trace(go.Bar(x=hist.index, y=mhist, marker_color=mc, opacity=0.7,
                         name="MACD Hist", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=macd_l, mode="lines",
        line=dict(color="#4299e1",width=1.5), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=sig_l, mode="lines",
        line=dict(color="#fc8181",width=1.5,dash="dot"), name="Sinyal"), row=4, col=1)
    fig.add_hline(y=0, line_color="rgba(100,100,100,0.35)", line_width=1, row=4, col=1)

    fig.update_layout(
        height=700, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(family="Inter,sans-serif",color="#8892a4",size=11),
        legend=dict(bgcolor="rgba(26,31,46,0.9)",bordercolor="#2d3748",
                    borderwidth=1,font=dict(size=11),x=0,y=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0,r=0,t=28,b=0), hovermode="x unified",
    )
    for r_ in [1,2,3,4]:
        fig.update_yaxes(gridcolor="#1a1f2e",zerolinecolor="#2d3748",tickfont=dict(size=10),row=r_,col=1)
        fig.update_xaxes(gridcolor="#1a1f2e",showticklabels=(r_==4),row=r_,col=1)
    fig.update_yaxes(range=[0,100],row=3,col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Sinyal özeti
    st.markdown("<div class='section-title'>⚡ Teknik Sinyal Özeti</div>", unsafe_allow_html=True)
    rsi_v=last_val(rsi); macd_v=last_val(mhist)
    s20=last_val(sma20); s50=last_val(sma50); s200=last_val(sma200)
    bbu=last_val(bb_u);  bbl=last_val(bb_l)

    signals=[]
    if rsi_v:
        if rsi_v>=70:   signals.append(("RSI",f"{rsi_v:.1f} — Aşırı Alım","sell"))
        elif rsi_v<=30: signals.append(("RSI",f"{rsi_v:.1f} — Aşırı Satım","buy"))
        else:           signals.append(("RSI",f"{rsi_v:.1f} — Nötr","hold"))
    if macd_v is not None:
        signals.append(("MACD","Pozitif Histogram — Alım" if macd_v>0 else "Negatif Histogram — Satış",
                         "buy" if macd_v>0 else "sell"))
    for ln_,sv_ in [("SMA 20",s20),("SMA 50",s50),("SMA 200",s200)]:
        if sv_ and not np.isnan(sv_):
            signals.append((ln_,f"{'Üstünde' if cur>sv_ else 'Altında'} ({sv_:.2f})",
                             "buy" if cur>sv_ else "sell"))
    if bbu and bbl:
        if cur>bbu:   signals.append(("Bollinger",f"Üst Band Üstünde ({bbu:.2f})","sell"))
        elif cur<bbl: signals.append(("Bollinger",f"Alt Band Altında ({bbl:.2f})","buy"))
        else:         signals.append(("Bollinger","Band İçinde — Nötr","hold"))

    bc=sum(1 for _,_,c in signals if c=="buy")
    sc=sum(1 for _,_,c in signals if c=="sell")
    hc=sum(1 for _,_,c in signals if c=="hold")
    genel="🟢 AL" if bc>sc else ("🔴 SAT" if sc>bc else "⚪ NÖTR")

    c1,c2,c3,c4=st.columns(4)
    for col_,l_,v_,cl_ in [(c1,"AL Sinyali",str(bc),"positive"),
                             (c2,"SAT Sinyali",str(sc),"negative"),
                             (c3,"NÖTR",str(hc),""),
                             (c4,"Genel Görünüm",genel,"")]:
        with col_: st.markdown(mcard(l_,v_,cl_), unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)
    st.markdown("".join(f'<span class="badge badge-{c}">● {n}: {d}</span>'
                        for n,d,c in signals), unsafe_allow_html=True)

    # Pivot
    if len(hist) >= 2:
        st.markdown("<div class='section-title'>📍 Pivot Noktaları (Önceki Gün)</div>", unsafe_allow_html=True)
        ph=float(high.iloc[-2]); pl=float(low.iloc[-2]); pc=float(close.iloc[-2])
        pv=(ph+pl+pc)/3
        levels=[("R3",ph+2*(pv-pl),"#fc8181"),("R2",pv+ph-pl,"#fc8181"),
                ("R1",2*pv-pl,"#feb2b2"),("Pivot",pv,"#f6e05e"),
                ("S1",2*pv-ph,"#9ae6b4"),("S2",pv-ph+pl,"#48bb78"),
                ("S3",pl-2*(ph-pv),"#276749")]
        pp=st.columns(7)
        for i,(l_,v_,c_) in enumerate(levels):
            with pp[i]:
                st.markdown(f'<div class="metric-box"><div class="metric-label">{l_}</div>'
                            f'<div style="font-size:14px;font-weight:700;color:{c_}">{v_:,.2f}</div></div>',
                            unsafe_allow_html=True)


def render_holders_tab(sym: str, is_us: bool = False):
    """Yabancı & Ortaklık sekmesi — ortak."""
    currency = "$" if is_us else "₺"

    st.markdown(f"""<div class="info-box">
        ℹ️ Veri kaynağı: Yahoo Finance · 
        {'ABD hisseleri için kurumsal sahiplik verileri genellikle eksiksizdir.' if is_us else
         'Bazı BIST hisseleri için veri sınırlı olabilir. Detay için '
         '<a href="https://www.kap.org.tr" target="_blank" style="color:#4299e1">kap.org.tr</a> veya '
         '<a href="https://www.isyatirim.com.tr" target="_blank" style="color:#4299e1">isyatirim.com.tr</a>'}
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title'>🌍 Pay Sahipliği Dağılımı</div>", unsafe_allow_html=True)
        mh = get_major_holders(sym, is_us)
        if mh:
            for lbl, val in mh:
                try:
                    pf = float(str(val).replace("%","").replace(",",".").strip())
                    bw = min(int(pf), 100)
                    bc = "#4299e1" if pf<40 else ("#48bb78" if pf<70 else "#fc8181")
                    st.markdown(f"""
                    <div style="margin-bottom:14px">
                        <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                            <span style="font-size:12px;color:#8892a4">{lbl}</span>
                            <span style="font-size:14px;font-weight:700;color:#e2e8f0">{pf:.2f}%</span>
                        </div>
                        <div style="background:#0e1117;border-radius:4px;height:7px">
                            <div style="width:{bw}%;background:{bc};height:7px;border-radius:4px"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                except Exception:
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1e2435">
                        <span style="font-size:12px;color:#8892a4">{lbl}</span>
                        <span style="font-size:13px;color:#e2e8f0;font-weight:600">{val}</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="info-box">
                📊 Pay sahipliği verisi mevcut değil.<br>
                <a href="https://www.kap.org.tr" target="_blank" style="color:#4299e1">KAP.org.tr</a> veya
                <a href="https://www.isyatirim.com.tr" target="_blank" style="color:#4299e1">İş Yatırım</a>
                üzerinden inceleyebilirsiniz.
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-title'>🏦 Kurumsal Yatırımcılar (İlk 10)</div>", unsafe_allow_html=True)
        ih = get_inst_holders(sym, is_us)
        if ih is not None and not ih.empty:
            # Sütun adlarını Türkçeleştir
            rename_map = {
                "Holder": "Kurum",
                "Shares": "Pay Adedi",
                "% Out":  "Hisse %",
                "Value":  f"Değer ({currency})",
                "Date Reported": "Rapor Tarihi",
            }
            cols_w = [c for c in rename_map if c in ih.columns]
            disp = ih[cols_w].head(10).copy().rename(columns=rename_map)

            if "Hisse %" in disp.columns:
                disp["Hisse %"] = disp["Hisse %"].apply(
                    lambda x: f"%{float(x)*100:.2f}" if pd.notna(x) else "—")
            if "Pay Adedi" in disp.columns:
                disp["Pay Adedi"] = disp["Pay Adedi"].apply(
                    lambda x: fmt_num(x,0) if pd.notna(x) else "—")
            if f"Değer ({currency})" in disp.columns:
                disp[f"Değer ({currency})"] = disp[f"Değer ({currency})"].apply(
                    lambda x: fmt_num(x,suffix=f" {currency}") if pd.notna(x) else "—")
            if "Rapor Tarihi" in disp.columns:
                disp["Rapor Tarihi"] = disp["Rapor Tarihi"].apply(
                    lambda x: x.strftime("%d.%m.%Y") if hasattr(x,"strftime") else str(x)[:10])

            st.dataframe(disp, use_container_width=True, hide_index=True, height=300)

            # Mini pasta — en büyük 5
            if "Kurum" in disp.columns and "Pay Adedi" in ih.columns:
                t5 = ih.head(5).dropna(subset=["Holder","Shares"])
                if not t5.empty:
                    fig_ih = go.Figure(go.Pie(
                        labels=t5["Holder"].str[:26],
                        values=t5["Shares"].astype(float),
                        hole=0.5, textfont=dict(size=10),
                        marker=dict(colors=["#4299e1","#48bb78","#9f7aea","#f6e05e","#fc8181"]),
                    ))
                    fig_ih.update_layout(height=210,margin=dict(l=0,r=0,t=6,b=0),
                        paper_bgcolor="#0e1117",font=dict(color="#8892a4",size=10),
                        legend=dict(font=dict(size=9),bgcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig_ih, use_container_width=True)
        else:
            st.markdown("""<div class="info-box">
                📊 Kurumsal yatırımcı verisi bulunamadı.<br>
                <a href="https://www.mkk.com.tr" target="_blank" style="color:#4299e1">MKK</a> veya
                <a href="https://www.isyatirim.com.tr" target="_blank" style="color:#4299e1">İş Yatırım</a>
                üzerinden inceleyebilirsiniz.
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>👤 İçeriden Alım-Satım</div>", unsafe_allow_html=True)
    try:
        ins = get_ticker(sym, is_us).insider_transactions
        if ins is not None and not ins.empty:
            ins_rename = {
                "Insider": "İçeriden", "Position": "Pozisyon",
                "Transaction": "İşlem", "Shares": "Pay",
                "Value": "Değer", "Start Date": "Tarih",
            }
            ic = [c for c in ins_rename if c in ins.columns]
            disp_ins = ins[ic].head(10).copy().rename(columns=ins_rename)
            st.dataframe(disp_ins, use_container_width=True, hide_index=True, height=240)
        else:
            st.info("İçeriden işlem verisi bulunamadı.")
    except Exception:
        st.info("İçeriden işlem verisi alınamadı.")


def render_analyst_tab(sym: str, info: dict, is_us: bool = False):
    """Analistler & Temettü sekmesi — ortak."""
    currency = "$" if is_us else "₺"
    cur_price = (info.get("currentPrice") or info.get("regularMarketPrice") or 0)

    col_an, col_dv = st.columns(2)

    with col_an:
        st.markdown("<div class='section-title'>🎯 Analist Tavsiyeleri</div>", unsafe_allow_html=True)
        rec = get_analyst_rec(sym, is_us)
        if rec is not None and not rec.empty:
            cc = [c for c in ["strongBuy","buy","hold","sell","strongSell"] if c in rec.columns]
            if cc:
                tot  = rec[cc].sum()
                lmap = {"strongBuy":"Güçlü Al","buy":"Al","hold":"Tut","sell":"Sat","strongSell":"Güçlü Sat"}
                cmap = {"strongBuy":"#276749","buy":"#48bb78","hold":"#d69e2e","sell":"#fc8181","strongSell":"#742a2a"}
                vlst = [float(tot[k]) for k in cc if float(tot[k])>0]
                llst = [lmap.get(k,k) for k in cc if float(tot[k])>0]
                clst = [cmap.get(k,"#4299e1") for k in cc if float(tot[k])>0]
                fig_p = go.Figure(go.Pie(labels=llst,values=vlst,hole=0.55,
                    marker_colors=clst,textfont=dict(size=12)))
                fig_p.update_layout(height=250,margin=dict(l=0,r=0,t=6,b=0),
                    paper_bgcolor="#0e1117",font=dict(color="#8892a4"),
                    legend=dict(font=dict(size=11),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_p, use_container_width=True)

                sc_ = st.columns(len(cc))
                for i,k in enumerate(cc):
                    with sc_[i]:
                        st.markdown(f'<div class="metric-box"><div class="metric-label">{lmap.get(k,k)}</div>'
                                    f'<div style="font-size:20px;font-weight:700;color:{cmap.get(k,"#e2e8f0")}">'
                                    f'{int(tot[k])}</div></div>', unsafe_allow_html=True)

            rd = rec.sort_values("period",ascending=False).head(8) if "period" in rec.columns else rec.tail(8)
            st.dataframe(rd, use_container_width=True, hide_index=True, height=170)
        else:
            st.info("Analist tavsiye verisi bulunamadı.")

        st.markdown("<div class='section-title'>💰 Fiyat Hedefleri</div>", unsafe_allow_html=True)
        tm = safe(info,"targetMeanPrice")
        th = safe(info,"targetHighPrice")
        tl = safe(info,"targetLowPrice")
        an = safe(info,"numberOfAnalystOpinions")
        rk = safe(info,"recommendationKey")

        if any(x != "—" for x in [tm,th,tl]):
            tc_ = st.columns(3)
            for col_,l_,v_ in [(tc_[0],"Ort. Hedef",tm),(tc_[1],"Yük. Hedef",th),(tc_[2],"Düş. Hedef",tl)]:
                with col_:
                    dv_ = f"{float(v_):,.2f} {currency}" if v_!="—" else "—"
                    try:
                        up = (float(v_)-cur_price)/cur_price*100
                        ups = f'<div style="font-size:11px;color:{"#48bb78" if up>=0 else "#fc8181"};margin-top:3px">{"▲" if up>=0 else "▼"}{abs(up):.1f}%</div>'
                    except Exception:
                        ups = ""
                    st.markdown(f'<div class="metric-box"><div class="metric-label">{l_}</div>'
                                f'<div style="font-size:15px;font-weight:700;color:#e2e8f0">{dv_}</div>{ups}</div>',
                                unsafe_allow_html=True)

        st.markdown(f"<table class='data-table'>"
                    f"<tr><td>Analist Sayısı</td><td>{an if an!='—' else '—'}</td></tr>"
                    f"<tr><td>Genel Tavsiye</td><td>{rk.upper() if rk!='—' else '—'}</td></tr>"
                    f"</table>", unsafe_allow_html=True)

    with col_dv:
        st.markdown("<div class='section-title'>💵 Temettü Geçmişi</div>", unsafe_allow_html=True)
        divs = get_dividends(sym, is_us)
        if divs is not None and not divs.empty:
            fig_dv = go.Figure(go.Bar(x=divs.index, y=divs.values,
                marker_color="#48bb78", opacity=0.8))
            fig_dv.update_layout(height=210,margin=dict(l=0,r=0,t=6,b=0),
                paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                xaxis=dict(gridcolor="#1a1f2e"),
                yaxis=dict(gridcolor="#1a1f2e",ticksuffix=f" {currency}"),
                font=dict(color="#8892a4",size=10))
            st.plotly_chart(fig_dv, use_container_width=True)
            ddf = divs.reset_index()
            ddf.columns = ["Tarih",f"Temettü ({currency})"]
            ddf["Tarih"] = ddf["Tarih"].dt.strftime("%d.%m.%Y")
            ddf[f"Temettü ({currency})"] = ddf[f"Temettü ({currency})"].map(lambda x: f"{x:.4f}")
            st.dataframe(ddf.sort_values("Tarih",ascending=False).head(15),
                         use_container_width=True, hide_index=True, height=240)
        else:
            st.info("Temettü geçmişi bulunamadı.")

        st.markdown("<div class='section-title'>📋 Kazanç Takvimi (EPS)</div>", unsafe_allow_html=True)
        try:
            earn = get_ticker(sym, is_us).earnings_dates
            if earn is not None and not earn.empty:
                earn_disp = earn.copy()
                if "EPS Estimate" in earn_disp.columns:
                    earn_disp = earn_disp.rename(columns={
                        "EPS Estimate": "EPS Tahmini",
                        "Reported EPS": "Açıklanan EPS",
                        "Surprise(%)": "Sürpriz %",
                    })
                st.dataframe(earn_disp.head(8), use_container_width=True, height=200)
            else:
                st.info("Kazanç takvimi verisi bulunamadı.")
        except Exception:
            st.info("Kazanç verisi alınamadı.")


# ═══════════════════════════════════════════════════════════════════════════════
# BIST HISSE BAŞLIĞI
# ═══════════════════════════════════════════════════════════════════════════════

def render_stock_header(sym, info, hist, is_us=False):
    currency = "$" if is_us else "₺"
    close = hist["Close"]
    cur   = float(close.iloc[-1])
    prev  = float(close.iloc[-2]) if len(close)>1 else cur
    chg   = cur - prev
    p     = (chg/prev*100) if prev!=0 else 0
    clr   = "#48bb78" if chg>=0 else "#fc8181"
    arr   = "▲" if chg>=0 else "▼"

    company  = safe(info,"longName","shortName",default=sym)
    sector   = safe(info,"sector",default="")
    industry = safe(info,"industry",default="")
    market   = "NYSE/NASDAQ" if is_us else "BIST"

    st.markdown(f"""
    <div class="stock-header">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
            <div>
                <div style="font-size:26px;font-weight:700;color:#e2e8f0">{company}</div>
                <div style="font-size:13px;color:#4299e1;font-weight:600;margin-top:3px">● {sym} · {market}</div>
                {f'<div style="font-size:12px;color:#718096;margin-top:3px">{sector} › {industry}</div>' if sector else ''}
            </div>
            <div style="text-align:right">
                <div style="font-size:32px;font-weight:700;color:{clr}">{cur:,.2f} {currency}</div>
                <div style="font-size:15px;font-weight:600;color:{clr}">{arr} {abs(chg):.2f} ({abs(p):.2f}%)</div>
                <div style="font-size:11px;color:#4a5568;margin-top:4px">{hist.index[-1].strftime('%d.%m.%Y')}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # 10 metrik
    pe    = safe(info,"trailingPE","forwardPE")
    pb    = safe(info,"priceToBook")
    mktcp = safe(info,"marketCap")
    ev    = safe(info,"enterpriseValue")
    roe   = safe(info,"returnOnEquity")
    d52h  = safe(info,"fiftyTwoWeekHigh")
    d52l  = safe(info,"fiftyTwoWeekLow")
    div_y = safe(info,"dividendYield")
    beta_ = safe(info,"beta")
    avg_v = safe(info,"averageVolume")

    cards = [
        ("F/K (P/E)",     fv(pe,".1f","x")),
        ("F/DD (P/B)",    fv(pb,".2f","x")),
        ("Piyasa Değeri", fmt_num(mktcp,suffix=f" {currency}")),
        ("EV",            fmt_num(ev,suffix=f" {currency}")),
        ("ROE",           pct(roe,x100=True)),
        (f"52H Yüksek",   fv(d52h,",.2f",f" {currency}")),
        (f"52H Düşük",    fv(d52l,",.2f",f" {currency}")),
        ("Temettü Verimi",pct(div_y,x100=True)),
        ("Beta",          fv(beta_,".2f")),
        ("Ort. Hacim",    fmt_num(avg_v,0)),
    ]
    r1 = st.columns(5); r2 = st.columns(5)
    for i,(l_,v_) in enumerate(cards):
        with (r1 if i<5 else r2)[i%5]:
            st.markdown(mcard(l_,v_), unsafe_allow_html=True)
    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Ana sekme: BIST mi ABD mi TEFAS mi
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
        <span style="font-size:28px">📈</span>
        <div>
            <div style="font-size:22px;font-weight:700;color:#e2e8f0">Hisse & Fon Analiz Paneli</div>
            <div style="font-size:12px;color:#718096">BIST · ABD Piyasaları · TEFAS Fonlar</div>
        </div>
    </div>""", unsafe_allow_html=True)

    main_tab1, main_tab2, main_tab3 = st.tabs([
        "🇹🇷 BIST Hisse",
        "🇺🇸 ABD Hisseleri",
        "💼 TEFAS Fonlar",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # BIST HISSE
    # ══════════════════════════════════════════════════════════════════════════
    with main_tab1:
        pmap = {"1ay":"1mo","3ay":"3mo","6ay":"6mo","1yıl":"1y","2yıl":"2y","5yıl":"5y"}
        bist_mode = st.radio("", ["🔍 Hisse Ara","📋 Hazır Listeler"], horizontal=True, key="bist_mode")

        if bist_mode == "🔍 Hisse Ara":
            cs, cp, cb = st.columns([3, 1.5, 0.8])
            with cs:
                bist_sym = st.text_input("", value="GARAN",
                    placeholder="GARAN · THYAO · ASELS · BIMAS…", label_visibility="collapsed",
                    key="bist_sym")
            with cp:
                bist_period = st.selectbox("", ["1ay","3ay","6ay","1yıl","2yıl","5yıl"],
                    index=2, label_visibility="collapsed", key="bist_period")
            with cb:
                bist_run = st.button("🔍 Analiz", use_container_width=True,
                                     type="primary", key="bist_run")

            if not bist_run and "bist_lt" not in st.session_state:
                st.markdown("""
                <div style="text-align:center;padding:50px;color:#4a5568">
                    <div style="font-size:44px;margin-bottom:12px">🇹🇷</div>
                    <div style="font-size:16px;color:#8892a4">Hisse kodu girin ve <b style="color:#4299e1">Analiz</b>'e tıklayın</div>
                    <div style="font-size:12px;margin-top:8px">GARAN · THYAO · ASELS · BIMAS · EREGL · SASA · AKBNK</div>
                </div>""", unsafe_allow_html=True)
            else:
                if bist_run:
                    st.session_state["bist_lt"] = bist_sym.strip().upper()
                    st.session_state["bist_lp"] = bist_period
                sym    = st.session_state.get("bist_lt", bist_sym.strip().upper())
                period = pmap[st.session_state.get("bist_lp", bist_period)]

                with st.spinner(f"**{sym}** yükleniyor…"):
                    try:
                        info, hist = get_info_and_hist(sym, period, is_us=False)
                    except Exception as e:
                        st.error(f"Veri alınamadı: {e}"); st.stop()

                if hist.empty:
                    st.error(f"**{sym}** için fiyat verisi bulunamadı."); st.stop()

                render_stock_header(sym, info, hist, is_us=False)
                t1, t2, t3 = st.tabs(["📊 Teknik Analiz","🌍 Yabancı & Ortaklık","🎯 Analistler & Temettü"])
                with t1: render_technical_tab(sym, period, is_us=False)
                with t2: render_holders_tab(sym, is_us=False)
                with t3: render_analyst_tab(sym, info, is_us=False)

        else:  # Hazır Listeler
            st.markdown("<div class='section-title'>📋 BIST Hisse Listesi Seç</div>", unsafe_allow_html=True)
            bcol1, bcol2 = st.columns([3, 1])
            with bcol1:
                bist_cat = st.selectbox("Kategori", list(BIST_PRESETS.keys()), key="bist_cat")
            with bcol2:
                bist_cnt = st.slider("Hisse Sayısı", 5, 30, 10, key="bist_cnt")

            bist_load = st.button("📥 Listeyi Yükle", type="primary", key="bist_load")

            tickers_bist = BIST_PRESETS[bist_cat][:bist_cnt]
            btags = "".join(f'<span class="tag">{t}</span>' for t in tickers_bist)
            st.markdown(f"<div style='margin:8px 0 12px 0'>{btags}</div>", unsafe_allow_html=True)

            if bist_load or f"bist_list_{bist_cat}" in st.session_state:
                if bist_load:
                    st.session_state[f"bist_list_{bist_cat}"] = True

                with st.spinner(f"**{len(tickers_bist)}** BIST hissesi yükleniyor…"):
                    df_bist = get_bist_screener_data(tuple(tickers_bist))

                if not df_bist.empty:
                    # Formatla
                    db = df_bist.copy()
                    if "Piyasa Değeri" in db.columns:
                        db["Piyasa Değeri"] = db["Piyasa Değeri"].apply(
                            lambda x: fmt_num(x, suffix=" ₺") if pd.notna(x) else "—")
                    if "Değ. %" in db.columns:
                        db["Değ. %"] = db["Değ. %"].apply(
                            lambda x: f"{float(x)*100:+.2f}%" if pd.notna(x) else "—")
                    if "ROE %" in db.columns:
                        db["ROE %"] = db["ROE %"].apply(
                            lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "—")
                    if "Temettü %" in db.columns:
                        db["Temettü %"] = db["Temettü %"].apply(
                            lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "—")
                    for nc in ["F/K","F/DD","Beta"]:
                        if nc in db.columns:
                            db[nc] = db[nc].apply(
                                lambda x: f"{float(x):.2f}" if pd.notna(x) and x!="" else "—")
                    for nc in ["Fiyat (₺)","52H Yük (₺)","52H Düş (₺)"]:
                        if nc in db.columns:
                            db[nc] = db[nc].apply(
                                lambda x: f"{float(x):,.2f}" if pd.notna(x) and x!="" else "—")

                    st.dataframe(db, use_container_width=True, hide_index=True,
                                 height=min(50 + len(db)*38, 550))

                    # Piyasa değeri bar grafik
                    df_bbar = df_bist.dropna(subset=["Piyasa Değeri","Sembol"]).head(15)
                    if not df_bbar.empty:
                        st.markdown("<div class='section-title'>📊 Piyasa Değeri Karşılaştırması</div>",
                                    unsafe_allow_html=True)
                        df_bbar = df_bbar.sort_values("Piyasa Değeri", ascending=True)
                        fig_bbar = go.Figure(go.Bar(
                            y=df_bbar["Sembol"],
                            x=df_bbar["Piyasa Değeri"] / 1e9,
                            orientation="h",
                            marker=dict(color=df_bbar["Piyasa Değeri"],
                                        colorscale="Teal", showscale=False),
                            text=df_bbar["Piyasa Değeri"].apply(lambda x: fmt_num(x,suffix=" ₺")),
                            textposition="outside",
                        ))
                        fig_bbar.update_layout(
                            height=max(280, len(df_bbar)*32),
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            xaxis=dict(title="Milyar ₺", gridcolor="#1a1f2e"),
                            yaxis=dict(gridcolor="#1a1f2e"),
                            font=dict(color="#8892a4", size=11),
                            margin=dict(l=0,r=90,t=10,b=0),
                        )
                        st.plotly_chart(fig_bbar, use_container_width=True)

                    # Detay analizi
                    st.markdown("<div class='section-title'>🔎 Detay İncele</div>", unsafe_allow_html=True)
                    bsel = st.selectbox("Hisse seç", df_bist["Sembol"].tolist(), key="bist_sel_detail")
                    if st.button("📈 Detay Analiz", key="bist_detail_btn"):
                        st.session_state["bist_detail_sym"] = bsel

                    if "bist_detail_sym" in st.session_state:
                        dsym = st.session_state["bist_detail_sym"]
                        with st.spinner(f"**{dsym}** detayları yükleniyor…"):
                            dinfo, dhist = get_info_and_hist(dsym, "6mo", is_us=False)
                        if not dhist.empty:
                            render_stock_header(dsym, dinfo, dhist, is_us=False)
                            bd1, bd2, bd3 = st.tabs(["📊 Teknik Analiz","🌍 Yabancı & Ortaklık","🎯 Analistler & Temettü"])
                            with bd1: render_technical_tab(dsym, "6mo", is_us=False)
                            with bd2: render_holders_tab(dsym, is_us=False)
                            with bd3: render_analyst_tab(dsym, dinfo, is_us=False)


    # ══════════════════════════════════════════════════════════════════════════
    # ABD HİSSELERİ
    # ══════════════════════════════════════════════════════════════════════════
    with main_tab2:
        us_mode = st.radio("", ["🔍 Hisse Ara","📋 Hazır Listeler"], horizontal=True, key="us_mode")

        pmap_us = {"1ay":"1mo","3ay":"3mo","6ay":"6mo","1yıl":"1y","2yıl":"2y","5yıl":"5y"}

        if us_mode == "🔍 Hisse Ara":
            cu1, cu2, cu3 = st.columns([3,1.5,0.8])
            with cu1:
                us_sym = st.text_input("", value="AAPL",
                    placeholder="AAPL · MSFT · NVDA · TSLA · AMZN…",
                    label_visibility="collapsed", key="us_sym")
            with cu2:
                us_period = st.selectbox("", ["1ay","3ay","6ay","1yıl","2yıl","5yıl"],
                    index=2, label_visibility="collapsed", key="us_period")
            with cu3:
                us_run = st.button("🔍 Analiz", use_container_width=True,
                                   type="primary", key="us_run")

            if not us_run and "us_lt" not in st.session_state:
                st.markdown("""
                <div style="text-align:center;padding:50px;color:#4a5568">
                    <div style="font-size:44px;margin-bottom:12px">🇺🇸</div>
                    <div style="font-size:16px;color:#8892a4">ABD hisse kodu girin</div>
                    <div style="font-size:12px;margin-top:8px">AAPL · MSFT · NVDA · GOOGL · AMZN · TSLA · META · JPM</div>
                </div>""", unsafe_allow_html=True)
            else:
                if us_run:
                    st.session_state["us_lt"] = us_sym.strip().upper()
                    st.session_state["us_lp"] = us_period
                sym    = st.session_state.get("us_lt", us_sym.strip().upper())
                period = pmap_us[st.session_state.get("us_lp", us_period)]

                with st.spinner(f"**{sym}** yükleniyor…"):
                    try:
                        info, hist = get_info_and_hist(sym, period, is_us=True)
                    except Exception as e:
                        st.error(f"Veri alınamadı: {e}"); st.stop()

                if hist.empty:
                    st.error(f"**{sym}** için fiyat verisi bulunamadı."); st.stop()

                render_stock_header(sym, info, hist, is_us=True)

                ut1, ut2, ut3 = st.tabs(["📊 Teknik Analiz","🌍 Kurumsal Sahiplik","🎯 Analistler & Temettü"])
                with ut1: render_technical_tab(sym, period, is_us=True)
                with ut2: render_holders_tab(sym, is_us=True)
                with ut3: render_analyst_tab(sym, info, is_us=True)

        else:  # Hazır Listeler
            st.markdown("<div class='section-title'>📋 Hisse Listesi Seç</div>", unsafe_allow_html=True)

            col_cat, col_cnt = st.columns([3, 1])
            with col_cat:
                category = st.selectbox("Kategori", list(US_PRESETS.keys()), key="us_cat")
            with col_cnt:
                count = st.slider("Hisse Sayısı", 5, 30, 15, key="us_cnt")

            load_list = st.button("📥 Listeyi Yükle", type="primary", key="us_load")

            # Seçili kategori etiketleri
            tickers_sel = US_PRESETS[category][:count]
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in tickers_sel)
            st.markdown(f"<div style='margin:8px 0 12px 0'>{tags_html}</div>", unsafe_allow_html=True)

            if load_list or f"us_list_{category}" in st.session_state:
                if load_list:
                    st.session_state[f"us_list_{category}"] = True

                with st.spinner(f"**{len(tickers_sel)}** hisse verisi yükleniyor… (bu biraz sürebilir)"):
                    df_screen = get_us_screener_data(tuple(tickers_sel))

                if not df_screen.empty:
                    # Sayısal sütunları formatla
                    df_disp = df_screen.copy()
                    if "Piyasa Değeri" in df_disp.columns:
                        df_disp["Piyasa Değeri"] = df_disp["Piyasa Değeri"].apply(
                            lambda x: fmt_num(x, suffix=" $") if pd.notna(x) else "—")
                    if "Değ. %" in df_disp.columns:
                        df_disp["Değ. %"] = df_disp["Değ. %"].apply(
                            lambda x: f"{float(x)*100:+.2f}%" if pd.notna(x) else "—")
                    if "ROE %" in df_disp.columns:
                        df_disp["ROE %"] = df_disp["ROE %"].apply(
                            lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "—")
                    if "Temettü %" in df_disp.columns:
                        df_disp["Temettü %"] = df_disp["Temettü %"].apply(
                            lambda x: f"{float(x)*100:.2f}%" if pd.notna(x) else "—")
                    for nc in ["F/K","F/DD","Beta"]:
                        if nc in df_disp.columns:
                            df_disp[nc] = df_disp[nc].apply(
                                lambda x: f"{float(x):.2f}" if pd.notna(x) and x!="" else "—")
                    for nc in ["Fiyat ($)","52H Yük ($)","52H Düş ($)"]:
                        if nc in df_disp.columns:
                            df_disp[nc] = df_disp[nc].apply(
                                lambda x: f"{float(x):,.2f}" if pd.notna(x) and x!="" else "—")

                    st.dataframe(df_disp, use_container_width=True, hide_index=True,
                                 height=min(50+len(df_disp)*38, 600))

                    # Piyasa değeri bar grafik
                    df_bar = df_screen.dropna(subset=["Piyasa Değeri","Sembol"]).head(15)
                    if not df_bar.empty:
                        st.markdown("<div class='section-title'>📊 Piyasa Değeri Karşılaştırması</div>",
                                    unsafe_allow_html=True)
                        df_bar = df_bar.sort_values("Piyasa Değeri", ascending=True)
                        fig_bar = go.Figure(go.Bar(
                            y=df_bar["Sembol"],
                            x=df_bar["Piyasa Değeri"] / 1e9,
                            orientation="h",
                            marker=dict(
                                color=df_bar["Piyasa Değeri"],
                                colorscale="Blues",
                                showscale=False,
                            ),
                            text=df_bar["Piyasa Değeri"].apply(lambda x: fmt_num(x,suffix=" $")),
                            textposition="outside",
                        ))
                        fig_bar.update_layout(
                            height=max(300, len(df_bar)*32),
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            xaxis=dict(title="Milyar $", gridcolor="#1a1f2e"),
                            yaxis=dict(gridcolor="#1a1f2e"),
                            font=dict(color="#8892a4",size=11),
                            margin=dict(l=0,r=80,t=10,b=0),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # Bir hisseye tıkla detay
                    st.markdown("<div class='section-title'>🔎 Detay İncele</div>", unsafe_allow_html=True)
                    sel_sym = st.selectbox("Hisse seç", df_screen["Sembol"].tolist(), key="us_sel_detail")
                    if st.button("📈 Detay Analiz", key="us_detail_btn"):
                        st.session_state["us_detail_sym"] = sel_sym

                    if "us_detail_sym" in st.session_state:
                        dsym = st.session_state["us_detail_sym"]
                        with st.spinner(f"**{dsym}** detayları yükleniyor…"):
                            dinfo, dhist = get_info_and_hist(dsym, "6mo", is_us=True)
                        if not dhist.empty:
                            render_stock_header(dsym, dinfo, dhist, is_us=True)
                            dt1, dt2 = st.tabs(["📊 Teknik Analiz","🎯 Analistler & Temettü"])
                            with dt1: render_technical_tab(dsym, "6mo", is_us=True)
                            with dt2: render_analyst_tab(dsym, dinfo, is_us=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TEFAS FONLAR  (tefas-crawler kütüphanesi kullanır)
    # ══════════════════════════════════════════════════════════════════════════
    with main_tab3:

        if not _tefas_crawler_available():
            st.error("**tefas-crawler** kütüphanesi bulunamadı.")
            st.code("pip install tefas-crawler", language="bash")
            st.stop()

        st.markdown("""
        <div class="info-box">
            💼 TEFAS fon kodu girerek NAV geçmişi, getiri tablosu, kıyaslama ve portföy dağılımı alın.<br>
            <span style="font-size:11px">Örnek kodlar: AFK · MAC · TI2 · GAF · YAF · IPB · NNF · KYD · GLD · AGF · GMF · TTE · AAK</span>
        </div>""", unsafe_allow_html=True)

        tf1, tf2 = st.columns([2, 1])
        with tf1:
            fund_code = st.text_input("Fon Kodu", value="MAC",
                placeholder="AFK · MAC · TI2 · GAF…", key="fund_code_main")
        with tf2:
            fund_days_sel = st.selectbox("Performans Periyodu",
                ["30 Gün","60 Gün","90 Gün","180 Gün","365 Gün"],
                index=2, key="fund_days_widget")
        load_fund = st.button("📂 Fon Yükle", type="primary", key="load_fund_btn")

        days_map = {"30 Gün":30,"60 Gün":60,"90 Gün":90,"180 Gün":180,"365 Gün":365}

        if not load_fund and "fund_loaded" not in st.session_state:
            st.markdown("""
            <div style="text-align:center;padding:40px;color:#4a5568">
                <div style="font-size:44px;margin-bottom:12px">💼</div>
                <div style="font-size:16px;color:#8892a4">Fon kodunu girin ve
                    <b style="color:#4299e1">Fon Yükle</b>'ye tıklayın</div>
                <div style="font-size:12px;margin-top:8px">
                    AFK · MAC · TI2 · GAF · YAF · KYD · GLD · AGF · GMF</div>
            </div>""", unsafe_allow_html=True)
        else:
            if load_fund:
                st.session_state["fund_loaded"]   = fund_code.strip().upper()
                st.session_state["fund_days_val"] = days_map[fund_days_sel]
            fc   = st.session_state.get("fund_loaded", fund_code.strip().upper())
            days = st.session_state.get("fund_days_val", 90)

            with st.spinner(f"**{fc}** TEFAS'tan yükleniyor…"):
                try:
                    df_hist  = tefas_fetch_history(fc, days)
                    latest_d = tefas_fetch_single_day(fc)
                except Exception as e:
                    st.error(f"Veri çekme hatası: {e}"); st.stop()

            if df_hist.empty:
                st.error(f"**{fc}** fon kodu bulunamadı ya da veri dönmedi.")
                st.stop()

            # ── Sütun normalize ───────────────────────────────────────────
            # tefas-crawler v0.5 sütunları:
            # date, price, code, title, market_cap, number_of_shares,
            # number_of_investors, stock, government_bond, eurobonds,
            # precious_metals, repo, reverse_repo, fx_payable_bills,
            # term_deposit, foreign_equity, etc.
            df = df_hist.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if "price" in df.columns:
                df["price"] = pd.to_numeric(df["price"], errors="coerce")
            if "market_cap" in df.columns:
                df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
            if "number_of_shares" in df.columns:
                df["number_of_shares"] = pd.to_numeric(df["number_of_shares"], errors="coerce")
            if "number_of_investors" in df.columns:
                df["number_of_investors"] = pd.to_numeric(df["number_of_investors"], errors="coerce")
            df = df.sort_values("date") if "date" in df.columns else df

            # ── Metrikler ─────────────────────────────────────────────────
            def _lv(col, default=0):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce").dropna()
                    return float(s.iloc[-1]) if not s.empty else default
                return float(latest_d.get(col, default) or default)

            nav_cur   = _lv("price")
            nav_old   = float(df["price"].dropna().iloc[0]) if "price" in df.columns and len(df["price"].dropna()) > 0 else nav_cur
            nav_chg   = ((nav_cur - nav_old) / nav_old * 100) if nav_old else 0
            nav_clr   = "#48bb78" if nav_chg >= 0 else "#fc8181"
            nav_arr   = "▲" if nav_chg >= 0 else "▼"
            mkt_cap   = _lv("market_cap")
            n_shares  = _lv("number_of_shares")
            n_inv     = _lv("number_of_investors")

            fon_adi = str(df["title"].dropna().iloc[-1]) if "title" in df.columns and not df["title"].dropna().empty else latest_d.get("title","—") or "—"
            fon_turu = str(df["fund_type"].dropna().iloc[-1]) if "fund_type" in df.columns and not df["fund_type"].dropna().empty else latest_d.get("fund_type","—") or "—"

            # ── Başlık ────────────────────────────────────────────────────
            st.markdown(f"""
            <div class="stock-header">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
                <div>
                  <div style="font-size:22px;font-weight:700;color:#e2e8f0">{fon_adi}</div>
                  <div style="font-size:13px;color:#4299e1;font-weight:600;margin-top:3px">● {fc} · TEFAS</div>
                  <div style="font-size:12px;color:#718096;margin-top:3px">Fon Türü: {fon_turu}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-size:28px;font-weight:700;color:{nav_clr}">{nav_cur:,.4f} ₺</div>
                  <div style="font-size:14px;font-weight:600;color:{nav_clr}">{nav_arr} {abs(nav_chg):.2f}% ({days} gün)</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── 4 metrik kart ─────────────────────────────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            ret_cls = "positive" if nav_chg >= 0 else "negative"
            for col_, lbl_, val_, cls_ in [
                (mc1, "Fon Toplam Değeri",  fmt_num(mkt_cap, suffix=" ₺"),  ""),
                (mc2, "Tedavül Pay Sayısı", fmt_num(n_shares, 0),           ""),
                (mc3, "Yatırımcı Sayısı",   fmt_num(n_inv, 0) if n_inv else "—", ""),
                (mc4, f"{days} Gün Getiri", f"%{nav_chg:+.2f}",            ret_cls),
            ]:
                with col_:
                    st.markdown(mcard(lbl_, val_, cls_), unsafe_allow_html=True)

            st.markdown("<div style='margin:10px 0'></div>", unsafe_allow_html=True)

            # ── 3 alt sekme ───────────────────────────────────────────────
            ft1, ft2, ft3 = st.tabs(["📈 NAV & Dönemsel Getiri", "📊 Kıyaslama", "🥧 Varlık Dağılımı"])

            # ─────────────────────────────────────────────────────────────
            # SEKME 1: NAV + Dönemsel Getiri
            # ─────────────────────────────────────────────────────────────
            with ft1:
                if "date" in df.columns and "price" in df.columns:
                    df_plot = df[["date","price"]].dropna()
                    if len(df_plot) > 1:
                        fig_nav = go.Figure()
                        fig_nav.add_trace(go.Scatter(
                            x=df_plot["date"], y=df_plot["price"],
                            mode="lines", name="Birim Pay Değeri",
                            line=dict(color="#4299e1", width=2.5),
                            fill="tozeroy", fillcolor="rgba(66,153,225,0.08)",
                        ))
                        fig_nav.update_layout(
                            height=320, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            xaxis=dict(gridcolor="#1a1f2e"),
                            yaxis=dict(gridcolor="#1a1f2e", ticksuffix=" ₺"),
                            font=dict(color="#8892a4", size=11),
                            margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                        )
                        st.plotly_chart(fig_nav, use_container_width=True)

                # Dönemsel getiri tablosu (günlük, haftalık, aylık, yıllık)
                st.markdown("<div class='section-title'>📋 Dönemsel Getiri Analizi</div>", unsafe_allow_html=True)
                if "price" in df.columns:
                    prices = df["price"].dropna().reset_index(drop=True)
                    cur_p  = float(prices.iloc[-1])
                    n      = len(prices)

                    perf_rows = []
                    periods = [
                        ("Günlük",    1),
                        ("Haftalık",  5),
                        ("Aylık",    21),
                        ("3 Aylık",  63),
                        ("6 Aylık", 126),
                        ("Yıllık",  252),
                        ("Tüm Dönem", n-1),
                    ]
                    for lbl_, d_ in periods:
                        if d_ < n:
                            old_p = float(prices.iloc[max(0, n-d_-1)])
                            ret   = (cur_p - old_p) / old_p * 100 if old_p else 0
                            icon  = "🟢" if ret >= 0 else "🔴"
                            perf_rows.append({
                                "Dönem":            lbl_,
                                "Başlangıç Fiyatı": f"{old_p:,.4f} ₺",
                                "Güncel Fiyat":     f"{cur_p:,.4f} ₺",
                                "Getiri":           f"{icon} {ret:+.2f}%",
                            })
                    if perf_rows:
                        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

                # Fon toplam değeri zaman serisi
                if "market_cap" in df.columns and df["market_cap"].dropna().shape[0] > 1:
                    st.markdown("<div class='section-title'>💰 Fon Toplam Değeri (AUM) Geçmişi</div>", unsafe_allow_html=True)
                    df_aum = df[["date","market_cap"]].dropna()
                    fig_aum = go.Figure(go.Scatter(
                        x=df_aum["date"], y=df_aum["market_cap"],
                        mode="lines", fill="tozeroy",
                        line=dict(color="#48bb78", width=2),
                        fillcolor="rgba(72,187,120,0.07)",
                        name="AUM",
                    ))
                    fig_aum.update_layout(
                        height=220, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        xaxis=dict(gridcolor="#1a1f2e"),
                        yaxis=dict(gridcolor="#1a1f2e",
                                   tickformat=".2s", ticksuffix=" ₺"),
                        font=dict(color="#8892a4", size=11),
                        margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                    )
                    st.plotly_chart(fig_aum, use_container_width=True)

                # Yatırımcı sayısı zaman serisi
                if "number_of_investors" in df.columns and df["number_of_investors"].dropna().shape[0] > 1:
                    st.markdown("<div class='section-title'>👥 Yatırımcı Sayısı Geçmişi</div>", unsafe_allow_html=True)
                    df_inv2 = df[["date","number_of_investors"]].dropna()
                    fig_inv = go.Figure(go.Scatter(
                        x=df_inv2["date"], y=df_inv2["number_of_investors"],
                        mode="lines", fill="tozeroy",
                        line=dict(color="#9f7aea", width=2),
                        fillcolor="rgba(159,122,234,0.07)",
                        name="Yatırımcı",
                    ))
                    fig_inv.update_layout(
                        height=200, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        xaxis=dict(gridcolor="#1a1f2e"),
                        yaxis=dict(gridcolor="#1a1f2e", tickformat=".2s"),
                        font=dict(color="#8892a4", size=11),
                        margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                    )
                    st.plotly_chart(fig_inv, use_container_width=True)

            # ─────────────────────────────────────────────────────────────
            # SEKME 2: Kıyaslama (Altın, Dolar, BIST100)
            # ─────────────────────────────────────────────────────────────
            with ft2:
                st.markdown("<div class='section-title'>📊 Kıyaslama: Fon vs Piyasa Göstergeleri</div>", unsafe_allow_html=True)
                st.markdown("""<div class="info-box" style="font-size:11px">
                    Tüm seriler başlangıç tarihine göre normalize edilmiştir (100 = başlangıç değeri).
                    Böylece yüzde bazlı karşılaştırma yapılabilir.</div>""", unsafe_allow_html=True)

                with st.spinner("Karşılaştırma verileri yükleniyor…"):
                    bench_data = get_comparison_data(days)

                if "date" in df.columns and "price" in df.columns:
                    df_cmp = df[["date","price"]].dropna().set_index("date")
                    df_cmp.index = pd.to_datetime(df_cmp.index)

                    # Normalize et (ilk değer = 100)
                    fig_cmp = go.Figure()

                    fund_norm = df_cmp["price"] / df_cmp["price"].iloc[0] * 100
                    fig_cmp.add_trace(go.Scatter(
                        x=fund_norm.index, y=fund_norm.values,
                        mode="lines", name=f"Fon ({fc})",
                        line=dict(color="#4299e1", width=2.5),
                    ))

                    bench_colors = {"BIST-100":"#48bb78","Dolar/TL":"#f6e05e",
                                    "Altın/TL":"#fbd38d","Gram Altın":"#ed8936"}
                    for label, series in bench_data.items():
                        try:
                            s = series.copy()
                            s.index = pd.to_datetime(s.index).tz_localize(None)
                            # Fon tarih aralığıyla hizala
                            s = s[s.index >= df_cmp.index.min()]
                            if len(s) < 2: continue
                            s_norm = s / s.iloc[0] * 100
                            fig_cmp.add_trace(go.Scatter(
                                x=s_norm.index, y=s_norm.values,
                                mode="lines", name=label,
                                line=dict(color=bench_colors.get(label,"#a0aec0"),
                                          width=1.8, dash="dot"),
                            ))
                        except Exception:
                            pass

                    fig_cmp.add_hline(y=100, line_color="rgba(100,100,100,0.4)",
                                      line_dash="dot", line_width=1)
                    fig_cmp.update_layout(
                        height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        xaxis=dict(gridcolor="#1a1f2e"),
                        yaxis=dict(gridcolor="#1a1f2e", ticksuffix=""),
                        font=dict(color="#8892a4", size=11),
                        legend=dict(bgcolor="rgba(26,31,46,0.9)", bordercolor="#2d3748",
                                    borderwidth=1, font=dict(size=11)),
                        margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    # Getiri karşılaştırma tablosu
                    cmp_rows = [{"Gösterge": f"🟦 Fon ({fc})",
                                 "Getiri (%)": f"{nav_chg:+.2f}%",
                                 "Son Değer": f"{nav_cur:,.4f} ₺"}]
                    for label, series in bench_data.items():
                        try:
                            s = series.copy()
                            s.index = pd.to_datetime(s.index).tz_localize(None)
                            s = s[s.index >= df_cmp.index.min()].dropna()
                            if len(s) < 2: continue
                            ret_b = (float(s.iloc[-1]) - float(s.iloc[0])) / float(s.iloc[0]) * 100
                            icon  = "🟢" if ret_b >= 0 else "🔴"
                            cmp_rows.append({
                                "Gösterge": label,
                                "Getiri (%)": f"{icon} {ret_b:+.2f}%",
                                "Son Değer": f"{float(s.iloc[-1]):,.2f}",
                            })
                        except Exception:
                            pass
                    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("Kıyaslama için fon fiyat verisi gerekiyor.")

            # ─────────────────────────────────────────────────────────────
            # SEKME 3: Varlık Dağılımı
            # ─────────────────────────────────────────────────────────────
            with ft3:
                # tefas-crawler'ın portföy sütunları (v0.5 şeması)
                # Hepsi 0-100 arası yüzde değeri
                ALLOC_COLS = {
                    "stock":                          "Hisse Senedi",
                    "government_bond":                "Devlet Tahvili",
                    "eurobonds":                      "Eurobond",
                    "precious_metals":                "Kıymetli Madenler",
                    "repo":                           "Repo",
                    "reverse_repo":                   "Ters Repo",
                    "term_deposit":                   "Vadeli Mevduat",
                    "fx_payable_bills":               "Döviz Ödemeli Bono",
                    "foreign_currency_bills":         "Dövizli Tahvil",
                    "bank_bills":                     "Banka Bonosu",
                    "exchange_traded_fund":           "Borsa Yatırım Fonu",
                    "fund_participation_certificate": "Fon Katılma Belgesi",
                    "commercial_paper":               "Finansman Bonosu",
                    "government_bonds_and_bills_fx":  "Kamu Dış Borç",
                    "participation_account":          "Katılım Hesabı",
                    "government_lease_certificates":  "Kamu Kira Sertif.",
                    "private_sector_lease_certificates": "Özel Sektör Kira",
                    "private_sector_bond":            "Özel Sektör Tahvil",
                    "derivatives":                    "Türev Araçlar",
                    "asset_backed_securities":        "Varlığa Dayalı Menkul",
                    "foreign_equity":                 "Yabancı Hisse",
                    "foreign_debt_instruments":       "Yabancı Borç Aracı",
                    "other":                          "Diğer",
                }

                # Mevcut sütunları filtrele
                avail = {v: df[k].dropna().iloc[-1]
                         for k, v in ALLOC_COLS.items()
                         if k in df.columns and not df[k].dropna().empty}
                avail = {k: float(v) for k, v in avail.items() if float(v) > 0.001}

                if avail:
                    labels  = list(avail.keys())
                    values  = list(avail.values())
                    total_w = sum(values)

                    col_pie, col_tbl = st.columns([1, 1])
                    with col_pie:
                        fig_fp = go.Figure(go.Pie(
                            labels=labels, values=values,
                            hole=0.45, textfont=dict(size=11),
                            textinfo="percent+label",
                        ))
                        fig_fp.update_layout(
                            height=380, margin=dict(l=0,r=0,t=16,b=0),
                            paper_bgcolor="#0e1117",
                            font=dict(color="#8892a4", size=10),
                            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
                            title=dict(text=f"Toplam: %{total_w:.1f}",
                                       font=dict(color="#8892a4", size=11),
                                       x=0.5),
                        )
                        st.plotly_chart(fig_fp, use_container_width=True)

                    with col_tbl:
                        # Zaman serisi bar: hangi varlık ne kadar ağırlıkta?
                        # Son değerleri bar olarak göster
                        df_bar_alloc = pd.DataFrame({
                            "Varlık Sınıfı": labels,
                            "Oran (%)": values,
                        }).sort_values("Oran (%)", ascending=True)

                        fig_hall = go.Figure(go.Bar(
                            y=df_bar_alloc["Varlık Sınıfı"],
                            x=df_bar_alloc["Oran (%)"],
                            orientation="h",
                            marker=dict(
                                color=df_bar_alloc["Oran (%)"],
                                colorscale="Viridis", showscale=False,
                            ),
                            text=df_bar_alloc["Oran (%)"].apply(lambda x: f"%{x:.2f}"),
                            textposition="outside",
                        ))
                        fig_hall.update_layout(
                            height=max(250, len(df_bar_alloc)*34),
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            xaxis=dict(gridcolor="#1a1f2e", ticksuffix="%"),
                            yaxis=dict(gridcolor="#1a1f2e", tickfont=dict(size=10)),
                            font=dict(color="#8892a4", size=11),
                            margin=dict(l=0, r=60, t=10, b=0),
                        )
                        st.plotly_chart(fig_hall, use_container_width=True)

                    # Varlık dağılımı zaman serisi (sadece büyük ağırlıklar)
                    top_alloc = sorted(avail.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_keys_raw = [k for k, _ in [(k2,v2) for k2, v2 in ALLOC_COLS.items()
                                                    if ALLOC_COLS.get(k2) in [t[0] for t in top_alloc]]
                                    if k in df.columns]
                    if len(top_keys_raw) > 1 and "date" in df.columns:
                        st.markdown("<div class='section-title'>📈 Varlık Dağılımı Değişimi</div>", unsafe_allow_html=True)
                        fig_ts = go.Figure()
                        COLORS = ["#4299e1","#48bb78","#f6e05e","#fc8181","#9f7aea","#ed8936"]
                        for i, k in enumerate(top_keys_raw):
                            label = ALLOC_COLS.get(k, k)
                            ser = pd.to_numeric(df[k], errors="coerce")
                            fig_ts.add_trace(go.Scatter(
                                x=df["date"], y=ser,
                                mode="lines", name=label,
                                line=dict(color=COLORS[i % len(COLORS)], width=1.8),
                            ))
                        fig_ts.update_layout(
                            height=260, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            xaxis=dict(gridcolor="#1a1f2e"),
                            yaxis=dict(gridcolor="#1a1f2e", ticksuffix="%"),
                            legend=dict(bgcolor="rgba(26,31,46,0.9)",
                                        bordercolor="#2d3748", borderwidth=1,
                                        font=dict(size=10)),
                            font=dict(color="#8892a4", size=11),
                            margin=dict(l=0,r=0,t=10,b=0), hovermode="x unified",
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("Bu fon için portföy dağılım verisi bulunamadı.")

    # Footer
    st.markdown("""
    <div style="text-align:center;margin-top:36px;padding:16px;border-top:1px solid #1e2435">
        <span style="font-size:11px;color:#4a5568">
            Veri: Yahoo Finance · TEFAS &nbsp;|&nbsp; ⚠️ Yatırım tavsiyesi değildir
        </span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
