import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title('Stock Market Dashboard')
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('start date')
end_date = st.sidebar.date_input('end date')

if ticker:
    data = yf.download(ticker, start=start_date, end=end_date)
    data.columns = data.columns.get_level_values(0)

    fig = px.line(data, x=data.index, y=data['Close'], title=ticker)
    st.plotly_chart(fig)

    pricing_data, fundamental_data, news, ml_model = st.tabs(['Pricing Data', 'Fundamental Data', 'News', 'ML Model'])

    with pricing_data:
        st.header("Pricing of the Stock")
        st.dataframe(data)
        data2 = data.copy()
        data2.dropna(inplace=True)
        data2['%Change'] = data2['Close'] / data2['Close'].shift(1) - 1

        st.write(data2)
        annual_return = round(data2['%Change'].mean() * 252 * 100, 2)
        st.write('Annual return is : ', annual_return, '%')

        stdev = round(np.std(data2['%Change']) * np.sqrt(252), 5)
        st.write("Standard deviation is", stdev * 100, '%')

    from alpha_vantage.fundamentaldata import FundamentalData
    with fundamental_data:
        key = 'Y8H8OOY0012QH5YM'
        fd = FundamentalData(key, output_format='pandas')

        st.subheader('Balance sheet')
        balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        st.write(bs)

        st.subheader('Income Statement')
        in_st = fd.get_income_statement_annual(ticker)[0]
        inst = in_st.T[2:]
        inst.columns = list(in_st.T.iloc[0])
        st.write(inst)

        st.subheader('Cash Flow Statement')
        cash_flow = fd.get_cash_flow_annual(ticker)[0]
        cf = cash_flow.T[2:]
        cf.columns = list(cash_flow.T.iloc[0])
        st.write(cf)

    from stocknews import StockNews
    with news:
        st.subheader(f"News of {ticker}")
        sn = StockNews(ticker, save_news=False)
        df_news = sn.read_rss()
        for i in range(10):
            st.subheader(f"News {i+1}")
            st.write(df_news['published'][i])
            st.write(df_news['title'][i])
            st.write(df_news['summary'][i])

            title_sentiment = df_news['sentiment_title'][i]
            st.write('Title sentiment is: ', title_sentiment)
            news_sentiment = df_news['sentiment_summary'][i]
            st.write('News statement sentiment:', news_sentiment)

        # Sentiment Aggregation
        df_news['date'] = pd.to_datetime(df_news['published']).dt.date
        daily_sentiment = df_news.groupby('date')[['sentiment_summary']].mean()
        st.line_chart(daily_sentiment)

    with ml_model:
        st.subheader("ML Model: Predict Stock Movement")
        data_ml = data[['Close']].copy()
        data_ml['Target'] = (data_ml['Close'].shift(-1) > data_ml['Close']).astype(int)
        data_ml['Close_prev'] = data_ml['Close'].shift(1)
        data_ml.dropna(inplace=True)

        X = data_ml[['Close_prev']]
        y = data_ml['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy of Movement Prediction Model: {round(accuracy*100, 2)}%")

        latest_close = data_ml['Close'].iloc[-1]
        predicted_movement = model.predict([[latest_close]])[0]
        if predicted_movement:
            st.success("The model predicts the stock will go UP tomorrow.")
        else:
            st.warning("The model predicts the stock will go DOWN tomorrow.")
