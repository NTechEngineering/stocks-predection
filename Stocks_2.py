import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
import plotly.graph_objects as go

# Download NLTK data quietly
nltk.download('vader_lexicon', quiet=True)

# --- Streamlit UI Configuration ---
st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar App inputs
st.sidebar.header("⚙️ Configuration")
ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., SOTL.NS, AAPL):", value="SOTL.NS").upper()
analyze_button = st.sidebar.button("Run Prediction Analysis", type="primary")

# Main Title
st.title("📈 AI-Powered Stock Movement Predictor")
st.markdown("This app uses a **Random Forest Regressor** paired with **VADER Sentiment Analysis** from recent news headlines to forecast tomorrow's closing price movement.")
st.write("---")

def predict_stock_movement(ticker_symbol):
    # Step 1: Fetching Data
    with st.spinner(f"Fetching historical market data for {ticker_symbol}..."):
        stock = yf.Ticker(ticker_symbol)
        df = stock.history(period="1y")
    
    if df.empty:
        st.error(f"❌ Error: No data found for ticker '{ticker_symbol}'. Please verify the symbol name.")
        return

    # Step 2: Sentiment Analysis
    with st.spinner("Analyzing recent news headlines sentiment..."):
        news = stock.news
        sentiment_scores = []
        sia = SentimentIntensityAnalyzer()
        
        if news:
            for article in news:
                title = article.get('title', '')
                score = sia.polarity_scores(title)['compound']
                sentiment_scores.append(score)
            avg_sentiment = np.mean(sentiment_scores)
        else:
            avg_sentiment = 0.0 

    # Step 3: Feature Engineering
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    df['Sentiment'] = avg_sentiment 
    df['Target'] = df['Close'].shift(-1)
    
    df.dropna(inplace=True)

    features = ['Close', 'Close_Lag1', 'Close_Lag2', 'Close_Lag3', 'Sentiment']
    X = df[features]
    y = df['Target']
    
    # Step 4: Model Training
    with st.spinner("Training Machine Learning Model (Random Forest)..."):
        train_size = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    # Step 5: Make Prediction
    latest_market_data = X.iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(latest_market_data)[0]
    
    last_actual_close = df['Close'].iloc[-1]
    price_difference = predicted_price - last_actual_close
    
    # Determine direction styling
    if price_difference > 0:
        direction = "UP 🔼"
        delta_color = "normal"
    else:
        direction = "DOWN 🔽"
        delta_color = "inverse"

    # --- UI RESULTS DISPLAY ---
    st.subheader(f"📊 Analysis Results for {ticker_symbol}")
    
    # Grid columns for KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Last Actual Close", value=f"₹{last_actual_close:.2f}")
    with col2:
        st.metric(label="Predicted Tomorrow's Price", value=f"₹{predicted_price:.2f}", delta=f"{price_difference:+.2f}", delta_color=delta_color)
    with col3:
        st.metric(label="Expected Market Signal", value=direction)
    with col4:
        # Coloring sentiment based on value
        sent_label = "Neutral"
        if avg_sentiment > 0.05: sent_label = "Positive 😊"
        elif avg_sentiment < -0.05: sent_label = "Negative 😟"
        st.metric(label="News Sentiment Score", value=f"{avg_sentiment:.2f}", delta=sent_label, delta_color="off")

    # Interactive Price Chart
    st.markdown("### 📉 Historical Closing Prices (1 Year)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Close Price', line=dict(color='#007BFF', width=2)))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        xaxis_title="Date",
        yaxis_title="Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show raw data option
    with st.expander("👁️ View Extracted Feature Dataframe"):
        st.dataframe(df[features + ['Target']].tail(10), use_container_width=True)

# Trigger analysis based on user actions
if analyze_button:
    predict_stock_movement(ticker_symbol)
else:
    st.info("💡 Click the 'Run Prediction Analysis' button in the sidebar to start fetching and analyzing data.")