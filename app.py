import streamlit as st

import yfinance as yf

import pandas as pd

import numpy as np

import plotly.graph_objects as go



st.set_page_config(page_title="Stock Decision Support App", layout="wide")



st.title("📈 Stock Decision Support App")

st.caption("Educational research tool — not financial advice.")



st.warning(

    "This app is for educational research only. "

    "It does not provide personalized financial advice, guaranteed predictions, or buy/sell recommendations."

)



# -----------------------------

# Tickers

# -----------------------------



@st.cache_data(ttl=86400)

def get_sp500_tickers():

    fallback = [

        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",

        "META", "TSLA", "JPM", "V", "UNH",

        "XOM", "WMT", "MA", "PG", "HD",

        "COST", "AVGO", "LLY", "NFLX", "AMD",

        "BAC", "KO", "PEP", "CRM", "ADBE",

        "CSCO", "ORCL", "INTC", "IBM", "QCOM",

        "T", "DIS", "NKE", "MCD", "ABT",

        "MRK", "PFE", "CVX", "BA", "CAT",

        "GE", "GS", "AXP", "SPY", "QQQ",

        "DIA", "IWM", "VOO", "VTI", "SCHD"

    ]



    try:

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

        tables = pd.read_html(url)

        df = tables[0]

        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()



        if len(tickers) == 0:

            return fallback, pd.DataFrame({"Symbol": fallback})



        return tickers, df



    except Exception:

        return fallback, pd.DataFrame({"Symbol": fallback})





# -----------------------------

# Load stock data

# -----------------------------



@st.cache_data(ttl=3600)

def load_stock_data(ticker, period="2y"):

    try:

        df = yf.download(

            ticker,

            period=period,

            auto_adjust=True,

            progress=False,

            threads=False

        )



        if df is None or df.empty:

            return pd.DataFrame()



        df = df.reset_index()



        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)



        if "Date" not in df.columns or "Close" not in df.columns:

            return pd.DataFrame()



        return df



    except Exception:

        return pd.DataFrame()





@st.cache_data(ttl=3600)

def load_fundamentals(ticker):

    try:

        info = yf.Ticker(ticker).info



        return {

            "sector": info.get("sector"),

            "industry": info.get("industry"),

            "market_cap": info.get("marketCap"),

            "pe_ratio": info.get("trailingPE"),

            "forward_pe": info.get("forwardPE"),

            "profit_margin": info.get("profitMargins"),

            "revenue_growth": info.get("revenueGrowth"),

            "debt_to_equity": info.get("debtToEquity"),

        }



    except Exception:

        return {

            "sector": None,

            "industry": None,

            "market_cap": None,

            "pe_ratio": None,

            "forward_pe": None,

            "profit_margin": None,

            "revenue_growth": None,

            "debt_to_equity": None,

        }





# -----------------------------

# Indicators

# -----------------------------



def add_indicators(df):

    df = df.copy()



    df["Return"] = df["Close"].pct_change()

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    df["MA200"] = df["Close"].rolling(200).mean()



    df["Return_1M"] = df["Close"].pct_change(21)

    df["Return_3M"] = df["Close"].pct_change(63)

    df["Volatility_20D"] = df["Return"].rolling(20).std() * np.sqrt(252)



    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)



    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()



    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))



    return df





def safe_number(value, default=0):

    try:

        if value is None or pd.isna(value):

            return default

        return float(value)

    except Exception:

        return default





# -----------------------------

# Scoring

# -----------------------------



def calculate_scores(latest, fundamentals):

    score = 0

    reasons = []



    close = safe_number(latest.get("Close"))

    ma20 = safe_number(latest.get("MA20"))

    ma50 = safe_number(latest.get("MA50"))

    ma200 = safe_number(latest.get("MA200"))

    rsi = safe_number(latest.get("RSI"))

    return_1m = safe_number(latest.get("Return_1M"))

    return_3m = safe_number(latest.get("Return_3M"))

    volatility = safe_number(latest.get("Volatility_20D"), 1)



    if close > ma20:

        score += 1

        reasons.append("Price is above MA20.")

    else:

        reasons.append("Price is below MA20.")



    if close > ma50:

        score += 1

        reasons.append("Price is above MA50.")

    else:

        reasons.append("Price is below MA50.")



    if close > ma200:

        score += 1

        reasons.append("Price is above MA200.")

    else:

        reasons.append("Price is below MA200.")



    if ma20 > ma50:

        score += 1

        reasons.append("Short-term trend is stronger than medium-term trend.")

    else:

        reasons.append("Short-term trend is not stronger than medium-term trend.")



    if 45 <= rsi <= 65:

        score += 2

        reasons.append("RSI is in a healthy range.")

    elif 30 <= rsi < 45:

        score += 1

        reasons.append("RSI is neutral but weaker.")

    elif rsi > 70:

        score -= 1

        reasons.append("RSI may be overbought.")

    else:

        reasons.append("RSI is weak or oversold.")



    if return_1m > 0:

        score += 1

        reasons.append("1-month return is positive.")

    else:

        reasons.append("1-month return is negative.")



    if return_3m > 0:

        score += 1

        reasons.append("3-month return is positive.")

    else:

        reasons.append("3-month return is negative.")



    if volatility < 0.25:

        risk = "Low"

        score -= 0

        reasons.append("Volatility is low.")

    elif volatility < 0.45:

        risk = "Medium"

        score -= 1

        reasons.append("Volatility is moderate.")

    else:

        risk = "High"

        score -= 2

        reasons.append("Volatility is high.")



    pe = fundamentals.get("pe_ratio")

    profit_margin = fundamentals.get("profit_margin")

    revenue_growth = fundamentals.get("revenue_growth")

    debt_to_equity = fundamentals.get("debt_to_equity")



    if pe is not None and pe > 0 and pe < 35:

        score += 1

        reasons.append("P/E ratio appears reasonable.")

    elif pe is not None and pe >= 35:

        reasons.append("P/E ratio is high.")



    if profit_margin is not None and profit_margin > 0.10:

        score += 1

        reasons.append("Profit margin is strong.")



    if revenue_growth is not None and revenue_growth > 0.05:

        score += 1

        reasons.append("Revenue growth is positive.")



    if debt_to_equity is not None and debt_to_equity < 150:

        score += 1

        reasons.append("Debt-to-equity appears manageable.")



    if risk == "High" and score < 5:

        label = "Avoid for Now"

    elif score >= 9:

        label = "Strong Watch"

    elif score >= 7:

        label = "Watch"

    elif score >= 5:

        label = "Neutral"

    else:

        label = "Weak / High Caution"



    return score, risk, label, reasons





def analyze_stock(ticker, period="2y"):

    df = load_stock_data(ticker, period)



    if df.empty or len(df) < 220:

        return None



    df = add_indicators(df).dropna()



    if df.empty:

        return None



    fundamentals = load_fundamentals(ticker)

    latest = df.iloc[-1]



    score, risk, label, reasons = calculate_scores(latest, fundamentals)



    return {

        "ticker": ticker,

        "df": df,

        "latest": latest,

        "fundamentals": fundamentals,

        "score": score,

        "risk": risk,

        "label": label,

        "reasons": reasons,

    }





# -----------------------------

# Backtest

# -----------------------------



def backtest_stock(ticker, period="5y", forward_days=30):

    df = load_stock_data(ticker, period)



    if df.empty or len(df) < 260:

        return pd.DataFrame()



    df = add_indicators(df).dropna()



    if df.empty:

        return pd.DataFrame()



    results = []



    for i in range(220, len(df) - forward_days):

        current = df.iloc[i]



        fake_fundamentals = {

            "pe_ratio": 25,

            "profit_margin": 0.12,

            "revenue_growth": 0.08,

            "debt_to_equity": 100,

        }



        score, risk, label, reasons = calculate_scores(current, fake_fundamentals)



        current_price = df["Close"].iloc[i]

        future_price = df["Close"].iloc[i + forward_days]

        future_return = (future_price / current_price) - 1



        results.append({

            "Date": df["Date"].iloc[i],

            "Ticker": ticker,

            "Label": label,

            "Score": score,

            "Risk": risk,

            "Current Price": current_price,

            "Future Price": future_price,

            "Forward Return": future_return,

            "Positive Return": future_return > 0,

        })



    return pd.DataFrame(results)





def summarize_backtest(bt):

    if bt.empty:

        return pd.DataFrame()



    summary = bt.groupby("Label").agg(

        Count=("Forward Return", "count"),

        Avg_Return=("Forward Return", "mean"),

        Median_Return=("Forward Return", "median"),

        Win_Rate=("Positive Return", "mean"),

    ).reset_index()



    return summary.sort_values("Avg_Return", ascending=False)





# -----------------------------

# Chart

# -----------------------------



def make_price_chart(df, ticker):

    fig = go.Figure()



    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], mode="lines", name="MA20"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], mode="lines", name="MA50"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA200"], mode="lines", name="MA200"))



    fig.update_layout(

        title=f"{ticker} Price Trend",

        xaxis_title="Date",

        yaxis_title="Price",

        height=500

    )



    return fig





# -----------------------------

# App

# -----------------------------



tickers, sp500_df = get_sp500_tickers()



st.sidebar.header("Settings")



period = st.sidebar.selectbox(

    "Historical period",

    ["1y", "2y", "5y"],

    index=1

)



selected_ticker = st.sidebar.selectbox(

    "Select stock",

    tickers

)



max_stocks = min(50, len(tickers))

st.sidebar.write(f"Scanning first {max_stocks} stocks.")



forward_days = st.sidebar.selectbox(

    "Backtest forward days",

    [10, 20, 30, 60, 90],

    index=2

)





tab1, tab2, tab3, tab4 = st.tabs([

    "Single Stock",

    "Stock Screener",

    "Backtest One Stock",

    "Stock List"

])





with tab1:

    result = analyze_stock(selected_ticker, period)



    if result is None:

        st.error("Not enough data available for this stock.")

    else:

        latest = result["latest"]

        fundamentals = result["fundamentals"]



        st.subheader(f"{selected_ticker} Decision Summary")



        c1, c2, c3, c4, c5 = st.columns(5)



        c1.metric("Price", f"${safe_number(latest['Close']):.2f}")

        c2.metric("RSI", f"{safe_number(latest['RSI']):.1f}")

        c3.metric("Volatility", f"{safe_number(latest['Volatility_20D']):.1%}")

        c4.metric("Score", result["score"])

        c5.metric("Label", result["label"])



        st.plotly_chart(make_price_chart(result["df"], selected_ticker), use_container_width=True)



        st.markdown("### Explanation")

        for r in result["reasons"]:

            st.write(f"- {r}")



        st.markdown("### Fundamentals")



        fund_df = pd.DataFrame({

            "Metric": [

                "Sector",

                "Industry",

                "Market Cap",

                "P/E Ratio",

                "Forward P/E",

                "Profit Margin",

                "Revenue Growth",

                "Debt to Equity",

            ],

            "Value": [

                fundamentals.get("sector"),

                fundamentals.get("industry"),

                fundamentals.get("market_cap"),

                fundamentals.get("pe_ratio"),

                fundamentals.get("forward_pe"),

                fundamentals.get("profit_margin"),

                fundamentals.get("revenue_growth"),

                fundamentals.get("debt_to_equity"),

            ]

        })



        st.dataframe(fund_df, use_container_width=True)





with tab2:

    st.subheader("Stock Screener")



    selected_tickers = tickers[:max_stocks]

    rows = []



    progress = st.progress(0)



    for idx, ticker in enumerate(selected_tickers):

        result = analyze_stock(ticker, period)



        if result is not None:

            latest = result["latest"]

            fundamentals = result["fundamentals"]



            rows.append({

                "Ticker": ticker,

                "Label": result["label"],

                "Score": result["score"],

                "Risk": result["risk"],

                "Price": round(safe_number(latest["Close"]), 2),

                "RSI": round(safe_number(latest["RSI"]), 1),

                "Volatility": round(safe_number(latest["Volatility_20D"]), 3),

                "1M Return": round(safe_number(latest["Return_1M"]), 3),

                "3M Return": round(safe_number(latest["Return_3M"]), 3),

                "Sector": fundamentals.get("sector"),

                "P/E": fundamentals.get("pe_ratio"),

                "Profit Margin": fundamentals.get("profit_margin"),

                "Revenue Growth": fundamentals.get("revenue_growth"),

            })



        progress.progress((idx + 1) / len(selected_tickers))



    if rows:

        screener = pd.DataFrame(rows).sort_values("Score", ascending=False)

        st.dataframe(screener, use_container_width=True)



        csv = screener.to_csv(index=False).encode("utf-8")



        st.download_button(

            "Download Screener Results",

            csv,

            "stock_screener_results.csv",

            "text/csv"

        )

    else:

        st.warning("No results found.")





with tab3:

    st.subheader(f"Backtest: {selected_ticker}")



    bt = backtest_stock(

        selected_ticker,

        period="5y",

        forward_days=forward_days

    )



    if bt.empty:

        st.error("Not enough data for backtesting.")

    else:

        summary = summarize_backtest(bt)



        st.markdown("### Backtest Summary by Label")

        st.dataframe(summary, use_container_width=True)



        st.markdown("### Backtest Details")

        st.dataframe(bt.tail(200), use_container_width=True)



        fig = go.Figure()



        fig.add_trace(go.Box(

            x=bt["Label"],

            y=bt["Forward Return"],

            name="Forward Return"

        ))



        fig.update_layout(

            title=f"{selected_ticker} Forward Returns by Decision Label",

            xaxis_title="Decision Label",

            yaxis_title=f"{forward_days}-Day Forward Return",

            height=500

        )



        st.plotly_chart(fig, use_container_width=True)





with tab4:

    st.subheader("Stock Universe")

    st.dataframe(sp500_df, use_container_width=True)