import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from sec_api import ExtractorApi, QueryApi
from textblob import TextBlob
import html
import unicodedata
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Initial Setup ---
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ESG & Financial Screener Pro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center; }
    .stMetric { background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; }
    .alert-success { background: linear-gradient(90deg, #4CAF50, #45a049); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .alert-danger { background: linear-gradient(90deg, #f44336, #d32f2f); color: white; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .update-indicator { position: fixed; top: 10px; right: 10px; background: #4CAF50; color: white; padding: 0.5rem 1rem; border-radius: 5px; z-index: 1000; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    for key, default in [('analysis_results', None), ('last_analysis_time', None), ('previous_metrics', None), ('update_counter', 0)]:
        if key not in st.session_state: st.session_state[key] = default

# --- Core API & Analysis Functions (Hardened for Reliability) ---

def clean_text(text):
    """Cleans text by handling HTML entities and normalizing unicode."""
    if not isinstance(text, str): return ""
    return unicodedata.normalize("NFKC", html.unescape(text))

def extract_risk_factors(extractor_api, url, form_type):
    """Extracts risk factors from SEC filings using ExtractorApi."""
    try:
        item_code = "1A" if form_type == "10-K" else "part2item1a" if form_type == "10-Q" else None
        if not item_code: return ""
        return clean_text(extractor_api.get_section(url, item_code, "text"))
    except Exception as e:
        st.warning(f"Failed to extract from {url[:50]}...: {e}")
        return ""

def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob, with robust error handling."""
    if not text or len(text.strip()) < 20: return {"polarity": 0, "subjectivity": 0, "sentiment": "neutral"}
    try:
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity, 
            "subjectivity": blob.sentiment.subjectivity, 
            "sentiment": "positive" if blob.sentiment.polarity > 0.1 else "negative" if blob.sentiment.polarity < -0.1 else "neutral"
        }
    except: return {"polarity": 0, "subjectivity": 0, "sentiment": "neutral"}

def get_company_filings(api_key, ticker, max_filings):
    """Fetches recent 10-K and 10-Q filings for a given ticker."""
    try:
        queryApi = QueryApi(api_key=api_key)
        end_date, start_date = datetime.now(), datetime.now() - timedelta(days=2 * 365)
        query = {
          "query": f"ticker:{ticker} AND formType:(\"10-K\" OR \"10-Q\") AND filedAt:[{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}]",
          "from": "0", "size": str(max_filings), "sort": [{"filedAt": {"order": "desc"}}]
        }
        return queryApi.get_filings(query).get('filings', [])
    except Exception as e:
        st.warning(f"Could not fetch filings for {ticker}: {e}")
        return []

@st.cache_data(ttl=600)
def get_company_data(ticker):
    """Safely fetches comprehensive company data from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        if hist.empty: return None
        news_text = "".join([f"{article.get('title', '')} {article.get('summary', '')} " for article in stock.news[:5]])
        return {
            'name': info.get('longName', ticker), 'sector': info.get('sector', 'Unknown'), 'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0), 'business_summary': info.get('longBusinessSummary', ''), 'news_text': news_text,
            'volatility': hist['Close'].pct_change().std() * np.sqrt(252),
            'momentum': (hist['Close'].iloc[-1] / hist['Close'].iloc[-21] - 1) if len(hist) > 21 else 0,
            'pe_ratio': info.get('trailingPE'), 'current_price': hist['Close'].iloc[-1],
            'beta': info.get('beta', 1.0), 'dividend_yield': info.get('dividendYield', 0),
            'employees': info.get('fullTimeEmployees', 0)
        }
    except:
        st.warning(f"Could not fetch yfinance data for {ticker}.")
        return None

# --- Main Processing & Calculation Logic ---

class ESGAnalyzer:
    """Calculates ESG and Risk scores based on proprietary logic."""
    def __init__(self):
        self.sector_benchmarks = {'Technology':{'E':72,'S':78,'G':82},
                                  'Healthcare':{'E':68,'S':82,'G':77},
                                  'Financial Services':{'E':62,'S':72,'G':87},
                                  'Consumer Cyclical':{'E':58,'S':68,'G':72},
                                  'Energy':{'E':42,'S':62,'G':67},
                                  'Utilities':{'E':52,'S':72,'G':77},
                                  'Unknown':{'E':55,'S':55,'G':55}}

    def calculate_esg_score(self, sentiment_polarity, company_info):
        """Calculates ESG scores adjusted by sentiment and company data."""
        sector = company_info.get('sector', 'Unknown')
        base_scores = self.sector_benchmarks.get(sector, self.sector_benchmarks['Unknown'])
        sentiment_multiplier = 1 + (sentiment_polarity * 0.25)
        scores = {k: min(100, max(20, v * sentiment_multiplier)) for k, v in base_scores.items()}
        try:
            if company_info.get('market_cap', 0) > 1e11: scores['G'] = min(100, scores['G'] * 1.05)
            if company_info.get('volatility', 0.3) < 0.2: scores.update({'S': min(100, scores['S'] * 1.03), 'G': min(100, scores['G'] * 1.03)})
        except: pass
        return scores

    def calculate_risk_score(self, company_data, esg_scores, sentiment_polarity):
        """Calculates a composite risk score from multiple factors."""
        try:
            vol_risk = min(30, company_data.get('volatility', 0) * 100)
            esg_risk = max(0, (80 - np.mean(list(esg_scores.values()))) * 0.3125)
            sent_risk = max(0, (0.5 - sentiment_polarity) * 20)
            pe = company_data.get('pe_ratio', 20) or 20
            fin_risk = 15 if (pe > 30 or pe < 5) else 10 if (pe > 25 or pe < 10) else 5
            mkt_risk = 10 if company_data.get('market_cap', 0) < 1e9 else 5 if company_data.get('market_cap', 0) < 1e10 else 0
            return min(100, vol_risk + esg_risk + sent_risk + fin_risk + mkt_risk)
        except: return 50

def process_company_analysis(ticker, analyzer, extractor_api, api_key, include_sec, max_filings):
    """Main function to process a single company's analysis."""
    company_data = get_company_data(ticker)
    if not company_data: return None

    news_text = f"{company_data.get('business_summary', '')} {company_data.get('news_text', '')}"
    news_sentiment = analyze_sentiment(news_text)
    news_esg_scores = analyzer.calculate_esg_score(news_sentiment['polarity'], company_data)

    result = {
        'Ticker': ticker, 'Company': company_data['name'], 'Sector': company_data['sector'],
        'Market Cap ($B)': company_data.get('market_cap', 0) / 1e9, 'Volatility': company_data.get('volatility', 0),
        'PE Ratio': company_data.get('pe_ratio'), 'Beta': company_data.get('beta', 1.0), 'Current Price': company_data.get('current_price', 0),
        'Environmental Score': news_esg_scores['E'], 'Social Score': news_esg_scores['S'], 'Governance Score': news_esg_scores['G'],
        'Overall ESG Score': np.mean(list(news_esg_scores.values())),
        'Sentiment Score': news_sentiment['polarity'], 'Sentiment Label': news_sentiment['sentiment'],
        'Subjectivity': news_sentiment['subjectivity'],
    }
    result['Risk Score'] = analyzer.calculate_risk_score(company_data, news_esg_scores, news_sentiment['polarity'])
    
    if include_sec and extractor_api:
        filings = get_company_filings(api_key, ticker, max_filings)
        combined_text = " ".join([extract_risk_factors(extractor_api, f.get('linkToFilingDetails'), f.get('formType')) for f in filings if f.get('linkToFilingDetails')])
        result['SEC Filings Count'] = len(filings)
        
        if combined_text.strip():
            sec_sentiment = analyze_sentiment(combined_text)
            sec_esg_scores = analyzer.calculate_esg_score(sec_sentiment['polarity'], company_data)
            sec_overall_esg = np.mean(list(sec_esg_scores.values()))
            
            result.update({
                'SEC Sentiment Score': sec_sentiment['polarity'], 'SEC Sentiment Label': sec_sentiment['sentiment'],
                'SEC Overall ESG Score': sec_overall_esg,
                'Combined ESG Score': result['Overall ESG Score'] * 0.4 + sec_overall_esg * 0.6,
                'Combined Sentiment Score': result['Sentiment Score'] * 0.4 + sec_sentiment['polarity'] * 0.6
            })
    return result

# --- UI & Display ---

def update_metrics_dynamically(df):
    """Updates and displays the main metric cards with delta values."""
    esg_col = 'Combined ESG Score' if 'Combined ESG Score' in df.columns else 'Overall ESG Score'
    sentiment_col = 'Combined Sentiment Score' if 'Combined Sentiment Score' in df.columns else 'Sentiment Score'
    # ✅ safer column selection
    if 'Combined ESG Score' in df.columns:
        esg_col = 'Combined ESG Score'
    else:
        esg_col = 'Overall ESG Score'

    if 'Combined Sentiment Score' in df.columns:
        sentiment_col = 'Combined Sentiment Score'
    else:
        sentiment_col = 'Sentiment Score'
    current = {'avg_esg': df[esg_col].mean(), 'avg_sentiment': df[sentiment_col].mean(), 'esg_leaders': len(df[df[esg_col] > 70]), 'avg_risk': df['Risk Score'].mean()}
    delta = st.session_state.previous_metrics and {k: current[k] - st.session_state.previous_metrics.get(k, 0) for k in current} or {k: 0 for k in current}
    st.session_state.previous_metrics = current
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio ESG Score", f"{current['avg_esg']:.1f}", f"{delta['avg_esg']:.1f}" if abs(delta['avg_esg']) > 0.1 else None)
    col2.metric("Avg Sentiment", f"{current['avg_sentiment']:.3f}", f"{delta['avg_sentiment']:.3f}" if abs(delta['avg_sentiment']) > 0.01 else None)
    col3.metric("ESG Leaders (>70)", f"{current['esg_leaders']}", f"{delta['esg_leaders']}" if delta['esg_leaders'] != 0 else None)
    col4.metric("Portfolio Risk", f"{current['avg_risk']:.1f}", f"{delta['avg_risk']:.1f}" if abs(delta['avg_risk']) > 0.1 else None, help="High > 60, Medium > 35")

def display_results(df):
    """Renders the main results dashboard with multiple tabs and visualizations."""
    st.success(f"Successfully analyzed {len(df)} companies at {st.session_state.last_analysis_time.strftime('%H:%M:%S')}")
    update_metrics_dynamically(df)
    
    # ---- Safe column selection
    include_sec = 'SEC Filings Count' in df.columns
    esg_col = 'Combined ESG Score' if 'Combined ESG Score' in df.columns else 'Overall ESG Score'
    sentiment_col = 'Combined Sentiment Score' if 'Combined Sentiment Score' in df.columns else 'Sentiment Score'
    
    tabs = st.tabs(
        ["ESG Overview", "Financials", "Portfolio View", "Rankings"] +
        (["SEC Filings"] if include_sec else []) +
        ["Detailed Data"]
    )
    
    # ---------------- ESG Overview ----------------
    with tabs[0]:
        c1, c2 = st.columns(2)
        if esg_col in df.columns and sentiment_col in df.columns:
            fig_esg_sent = px.scatter(
                df, x=esg_col, y=sentiment_col,
                size='Market Cap ($B)', color='Sector',
                hover_data=['Ticker', 'Company', 'Risk Score'],
                title="ESG vs. Sentiment"
            )
            fig_esg_sent.update_layout(height=500).add_hline(y=0, line_dash="dash").add_vline(x=60, line_dash="dash")
            c1.plotly_chart(fig_esg_sent, use_container_width=True)
        
        if 'Volatility' in df.columns and 'Risk Score' in df.columns:
            fig_risk_vol = px.scatter(
                df, x='Volatility', y='Risk Score',
                color=esg_col if esg_col in df.columns else None,
                size='Market Cap ($B)', hover_data=['Ticker'],
                title="Risk vs. Volatility", color_continuous_scale='RdYlGn_r'
            )
            c2.plotly_chart(fig_risk_vol, use_container_width=True)
    
    # ---------------- Financials ----------------
    with tabs[1]:
        st.subheader("Key Financial Ratios")
        financial_cols = [col for col in ['Ticker', 'PE Ratio', 'Volatility', 'Beta'] if col in df.columns]
        if len(financial_cols) >= 2:  # needs at least Ticker + 1 metric
            df_financials = df[financial_cols].melt(id_vars='Ticker', var_name='Metric', value_name='Value')
            fig_radar = px.line_polar(
                df_financials, r='Value', theta='Metric', line_close=True, color='Ticker',
                title="Financial Ratios Comparison"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Financial ratio data is incomplete, so the radar chart cannot be displayed.")
    
    # ---------------- Portfolio View ----------------
    with tabs[2]:
        st.subheader("Portfolio Composition & ESG Heatmap")
        c1, c2 = st.columns(2)
        
        if 'Market Cap ($B)' in df.columns:
            df_treemap = df.copy()
            c1.plotly_chart(
                px.treemap(
                    df_treemap, path=[px.Constant("All Companies"), 'Sector', 'Ticker'],
                    values='Market Cap ($B)', color=sentiment_col if sentiment_col in df.columns else None,
                    title="Market Cap & Sentiment Treemap", color_continuous_scale='RdBu'
                ),
                use_container_width=True
            )
        
        esg_parts = [col for col in ['Environmental Score', 'Social Score', 'Governance Score'] if col in df.columns]
        if esg_parts:
            heatmap_data = df.set_index('Ticker')[esg_parts].round(1)
            c2.plotly_chart(
                px.imshow(
                    heatmap_data, text_auto=True, aspect="auto",
                    color_continuous_scale='Greens', title="ESG Component Score Heatmap"
                ),
                use_container_width=True
            )
    
    # ---------------- Rankings ----------------
    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            if esg_col in df.columns and sentiment_col in df.columns:
                st.markdown("#### Top ESG Performers")
                for rank, (idx, r) in enumerate(df.nlargest(5, esg_col).iterrows(), 1):
                    st.markdown(
                        f"<div class='alert-success'><strong>#{rank} {r['Ticker']}</strong>"
                        f"<br>ESG: {r[esg_col]:.1f} | Sentiment: {r[sentiment_col]:.3f}</div>",
                        unsafe_allow_html=True
                    )
        with c2:
            if 'Risk Score' in df.columns:
                st.markdown("#### High Risk Alerts")
                for rank, (idx, r) in enumerate(df.nlargest(5, 'Risk Score').iterrows(), 1):
                    risk_level = 'HIGH' if r['Risk Score'] > 70 else ('MEDIUM' if r['Risk Score'] > 35 else 'LOW')
                    st.markdown(
                        f"<div class='alert-danger'><strong>#{rank} {r['Ticker']}</strong>"
                        f"<br>Risk: {r['Risk Score']:.1f} ({risk_level})</div>",
                        unsafe_allow_html=True
                    )
    
    # ---------------- SEC Filings (Optional) ----------------
    tab_offset = 4
    if include_sec and 'SEC Overall ESG Score' in df.columns:
        with tabs[tab_offset]:
            sec_df = df.dropna(subset=['SEC Overall ESG Score'])
            if not sec_df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    numeric_df = sec_df[['Overall ESG Score', 'SEC Overall ESG Score']].select_dtypes(include=np.number)
                    if not numeric_df.empty:
                        min_val, max_val = numeric_df.min().min(), numeric_df.max().max()
                        fig_comp = px.scatter(
                            sec_df, x='Overall ESG Score', y='SEC Overall ESG Score',
                            color='Sector', hover_data=['Ticker'],
                            title="News ESG vs SEC Filings ESG"
                        )
                        fig_comp.add_shape(
                            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                            line=dict(color="red", dash="dash")
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
                with c2:
                    if 'SEC Sentiment Score' in sec_df.columns and 'Sentiment Score' in sec_df.columns:
                        fig_sent = px.scatter(
                            sec_df, x='Sentiment Score', y='SEC Sentiment Score',
                            color=esg_col if esg_col in sec_df.columns else None,
                            hover_data=['Ticker'], title="News Sentiment vs SEC Filings Sentiment"
                        )
                        fig_sent.add_shape(
                            type="line", x0=-1, y0=-1, x1=1, y1=1,
                            line=dict(color="red", dash="dash")
                        )
                        st.plotly_chart(fig_sent, use_container_width=True)
            else:
                st.info("No companies in the filtered set have SEC filings data.")
        tab_offset += 1
    
    # ---------------- Detailed Data ----------------
    with tabs[tab_offset]:
        st.subheader("Complete Analysis Dataset")
        df_csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            data=df_csv,
            file_name=f"esg_analysis_{datetime.now():%Y%m%d}.csv",
            mime="text/csv"
        )
        # Safe formatting with fallback
        style_cols = [c for c in [esg_col, sentiment_col] if c in df.columns]
        risk_cols = ['Risk Score'] if 'Risk Score' in df.columns else []
        try:
            styled_df = df.style.format(precision=2)
            if style_cols:
                styled_df = styled_df.background_gradient(cmap='RdYlGn', subset=style_cols)
            if risk_cols:
                styled_df = styled_df.background_gradient(cmap='RdYlGn_r', subset=risk_cols)
            st.dataframe(styled_df)
        except ImportError:
            # Fallback if matplotlib is not available
            st.dataframe(df.style.format(precision=2))
def create_esg_dashboard():
    """Main function to create the Streamlit dashboard UI and trigger analysis."""
    initialize_session_state()
    st.markdown('<div class="main-header"><h1>ESG & Financial Screener Pro</h1></div>', unsafe_allow_html=True)
    if st.session_state.update_counter > 0: st.markdown(f'<div class="update-indicator">Refreshed: {st.session_state.update_counter}</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Configuration")
        tickers_input = st.text_area("Tickers (comma-separated):", "AAPL,MSFT,GOOGL,TSLA,JPM,NVDA,V,UNH")
        include_sec = st.checkbox("Include SEC Filings Analysis", True)
        api_key_input = st.text_input("SEC API Key", type="password", value=os.getenv("SEC_API_KEY", ""))
        max_filings = st.slider("Max Filings per Ticker", 1, 10, 5)
        
        st.subheader("Display Filters")
        min_esg = st.slider("Minimum ESG Score", 0, 100, 0)
        min_sentiment = st.slider("Minimum Sentiment", -1.0, 1.0, -1.0, 0.1)
        min_mkt_cap = st.number_input("Minimum Market Cap ($B)", 0.0, value=0.0)

        if st.button("Run Analysis", type="primary", use_container_width=True):
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            if not tickers: st.error("Please enter at least one ticker."); return
            
            extractor_api = ExtractorApi(api_key_input) if include_sec and api_key_input else None
            if include_sec and not api_key_input: st.error("SEC analysis selected, but API key is missing."); return

            analyzer = ESGAnalyzer()
            results, start_time = [], time.time()
            progress_bar, status_text = st.progress(0), st.empty()
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ticker = {executor.submit(process_company_analysis, t, analyzer, extractor_api, api_key_input, include_sec, max_filings): t for t in tickers}
                for i, future in enumerate(as_completed(future_to_ticker), 1):
                    status_text.text(f"Analyzing {future_to_ticker[future]}... ({i}/{len(tickers)})")
                    try:
                        if data := future.result(): results.append(data)
                    except Exception as e: st.error(f"Critical error with {future_to_ticker[future]}: {e}")
                    progress_bar.progress(i / len(tickers))
            
            status_text.empty(); progress_bar.empty()
            
            if results:
                df = pd.DataFrame(results)
                # Final cleanup for numeric and categorical columns
                for col in df.columns:
                    if 'Score' in col: df[col] = pd.to_numeric(df[col].fillna(0))
                    if 'Label' in col: df[col] = df[col].fillna('neutral')
                
                esg_col = 'Combined ESG Score' if 'Combined ESG Score' in df.columns else 'Overall ESG Score'
                sent_col = 'Combined Sentiment Score' if 'Combined Sentiment Score' in df.columns else 'Sentiment Score'
                df_filtered = df[(df[esg_col] >= min_esg) & (df[sent_col] >= min_sentiment) & (df['Market Cap ($B)'] >= min_mkt_cap)]

                st.session_state.analysis_results = df_filtered if not df_filtered.empty else df
                if df_filtered.empty: st.warning("No companies passed filters. Showing all results.")
                
                st.session_state.last_analysis_time = datetime.now()
                st.session_state.update_counter += 1
                st.rerun()
            else: st.error("No data could be retrieved. Check tickers and API key.")

    if st.session_state.analysis_results is not None:
        display_results(st.session_state.analysis_results)

if __name__ == "__main__":
    create_esg_dashboard()