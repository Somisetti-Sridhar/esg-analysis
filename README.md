# 🌱 ESG & Financial Screener Pro

A comprehensive Streamlit application for analyzing Environmental, Social, and Governance (ESG) metrics alongside financial data for publicly traded companies. This tool combines real-time market data, sentiment analysis, and SEC filings to provide actionable insights for sustainable investing.

## Features

- **Multi-Source ESG Analysis**: Combines news sentiment, business summaries, and SEC filings
- **Real-Time Financial Data**: Fetches live market data via Yahoo Finance
- **Sentiment Analysis**: Uses TextBlob for natural language processing of company communications
- **SEC Filings Integration**: Analyzes 10-K and 10-Q risk factors using SEC API
- **Interactive Visualizations**: Built with Plotly for dynamic charts and dashboards
- **Risk Scoring**: Proprietary algorithm combining volatility, ESG, sentiment, and financial metrics
- **Sector Benchmarking**: Industry-specific ESG baseline scores
- **Portfolio View**: Treemap visualization of market cap and sentiment distribution
- **Concurrent Processing**: Multi-threaded analysis for faster results

## Installation

### Prerequisites

- Python 3.8 or higher
- SEC API key (get one at [sec-api.io](https://sec-api.io/))

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your SEC API key to `.env`:
```
SEC_API_KEY=your_actual_api_key_here
```

4. Download TextBlob corpora (first-time setup):
```bash
python -m textblob.download_corpora
```

## Usage

Run the Streamlit application:
```bash
streamlit run esg_v5.py
```

The app will open in your default browser at `http://localhost:8501`

### Configuration Options

**Sidebar Controls:**
- **Tickers**: Enter comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL)
- **Include SEC Filings**: Toggle SEC 10-K/10-Q analysis
- **SEC API Key**: Auto-loaded from `.env` file
- **Max Filings**: Number of recent filings to analyze per company (1-10)
- **Display Filters**:
  - Minimum ESG Score (0-100)
  - Minimum Sentiment (-1.0 to 1.0)
  - Minimum Market Cap (in billions)

### Dashboard Tabs

1. **📊 ESG Overview**: ESG vs Sentiment scatter plot, Risk vs Volatility analysis
2. **💰 Financials**: Radar chart comparing PE ratios, volatility, and beta
3. **📈 Portfolio View**: Market cap treemap and ESG component heatmap
4. **🏆 Rankings**: Top ESG performers and high-risk alerts
5. **📄 SEC Filings**: Comparison of news-based vs filing-based ESG scores
6. **📋 Detailed Data**: Full dataset with CSV download option

## How It Works

### ESG Scoring Methodology

The application uses a multi-layered approach:

1. **Sector Benchmarks**: Base ESG scores vary by industry (e.g., Technology: E=72, S=78, G=82)
2. **Sentiment Adjustment**: Scores are modified by sentiment polarity (±25%)
3. **Company Factors**: 
   - Large-cap companies (>$100B) receive governance boost
   - Low volatility (<20%) improves social and governance scores
4. **SEC Integration**: When enabled, combines news-based (40%) and filing-based (60%) scores

### Risk Score Components

- **Volatility Risk** (max 30 points): Based on annualized price volatility
- **ESG Risk** (max 25 points): Inverse relationship with ESG scores
- **Sentiment Risk** (max 10 points): Negative sentiment increases risk
- **Financial Risk** (max 15 points): Extreme P/E ratios flag valuation concerns
- **Market Risk** (max 10 points): Small-cap premium

## Security Best Practices

- **Never commit `.env` files**: Already included in `.gitignore`
- **Use environment variables**: API keys are loaded from `.env` via `python-dotenv`
- **Rotate API keys regularly**: Update your SEC API key periodically
- **Limit API exposure**: The key is masked in the UI with `type="password"`

## Data Sources

- **Yahoo Finance** (`yfinance`): Stock prices, financial ratios, company info, news
- **SEC EDGAR** (`sec-api`): Official 10-K and 10-Q filings
- **TextBlob**: Sentiment analysis and natural language processing

## Performance Notes

- Uses `ThreadPoolExecutor` with 8 workers for parallel processing
- Results are cached for 10 minutes using `@st.cache_data`
- Progress indicators show real-time analysis status
- Typical analysis time: 2-5 seconds per company

## Troubleshooting

**"Could not fetch yfinance data"**
- Check ticker symbols are valid and traded on major exchanges
- Verify internet connection

**"Failed to extract from SEC filing"**
- Ensure SEC API key is valid and has remaining quota
- Some filings may not have extractable risk factors

**"No data could be retrieved"**
- Verify at least one ticker is entered
- Check API key is properly set in `.env`

## Contributing

Contributions are welcome! Please ensure:
- Code follows PEP 8 style guidelines
- New features include appropriate error handling
- API keys remain secure and never hardcoded

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice. ESG scores are calculated using proprietary algorithms and should not be the sole basis for investment decisions. Always conduct thorough due diligence and consult with financial professionals.
