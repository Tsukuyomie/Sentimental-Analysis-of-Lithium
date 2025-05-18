import pandas as pd
from sqlalchemy import create_engine, inspect
import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Download the VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# SQL Server connection details
server_name = "14k"
database_name = "Sentimental_Analysis"
port = "1433"

# Create connection string for Windows Authentication
DATABASE_URL = f'mssql+pyodbc://{server_name}/{database_name}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'

# Create the engine
engine = create_engine(DATABASE_URL)

# Function to list all tables
def list_tables():
    inspector = inspect(engine)
    return inspector.get_table_names()

# Function to fetch Lithium Price data
def fetch_lithium_prices():
    try:
        price_query = "SELECT Dates, Price FROM LithiumPrice ORDER BY Dates DESC"
        price_df = pd.read_sql(price_query, engine)
        price_df['Dates'] = pd.to_datetime(price_df['Dates'], errors='coerce')
        return price_df
    except Exception as e:
        st.error(f"Error fetching Lithium Price data: {str(e)}")
        return pd.DataFrame()

# Function to fetch data from SQL Server with filtering
def fetch_data_from_db(months=6):
    tables = list_tables()
    st.write(f"Available tables in database: {tables}")
    
    table_name = None
    for table in tables:
        if "lithium" in table.lower() or "news" in table.lower():
            table_name = table
            break
    
    if not table_name:
        if tables:
            table_name = tables[0]
        else:
            st.error("No tables found in the database")
            return pd.DataFrame()
    
    inspector = inspect(engine)
    columns = [column['name'] for column in inspector.get_columns(table_name)]
    st.write(f"Columns in {table_name}: {columns}")
    
    title_col = next((col for col in columns if col.lower() == 'title'), None)
    links_col = next((col for col in columns if col.lower() in ['links', 'link', 'url']), None)
    source_col = next((col for col in columns if col.lower() == 'source'), None)
    date_col = next((col for col in columns if col.lower() in ['dates', 'date', 'datetime']), None)
    text_col = next((col for col in columns if col.lower() in ['texts', 'text', 'content']), None)
    
    existing_cols = []
    column_mapping = {}
    
    if title_col:
        existing_cols.append(title_col)
        column_mapping[title_col] = 'Title'
    if links_col:
        existing_cols.append(links_col)
        column_mapping[links_col] = 'Links'
    if source_col:
        existing_cols.append(source_col)
        column_mapping[source_col] = 'Source'
    if date_col:
        existing_cols.append(date_col)
        column_mapping[date_col] = 'Date'
    if text_col:
        existing_cols.append(text_col)
        column_mapping[text_col] = 'Text'
    
    cutoff_date = datetime(2024, 5, 1) - timedelta(days=30 * months)
    cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
    columns_str = ", ".join(existing_cols)
    
    if date_col:
        query = f"SELECT {columns_str} FROM {table_name} WHERE {date_col} >= '{cutoff_date_str}' ORDER BY {date_col} DESC"
    else:
        query = f"SELECT {columns_str} FROM {table_name}"
    
    st.write("Executing query:")
    st.code(query, language='sql')
    
    try:
        df = pd.read_sql(query, engine)
        df = df.rename(columns=column_mapping)
        return df
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        query = f"SELECT {columns_str} FROM {table_name}"
        st.write(f"Trying simpler query: {query}")
        df = pd.read_sql(query, engine)
        df = df.rename(columns=column_mapping)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'] >= cutoff_date]
        return df

# Function to analyze sentiment using VADER
def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()
    
    if 'Text' not in df.columns:
        st.error("Text column not found in data")
        return pd.DataFrame()
    
    result = []
    for index, row in df.iterrows():
        text = row['Text']
        sentiment_score = sid.polarity_scores(text)['compound']
        
        if sentiment_score >= 0.05:
            impact = "Positive"
        elif sentiment_score <= -0.05:
            impact = "Negative"
        else:
            impact = "Neutral"
        
        item = {
            "Title": row['Title'] if 'Title' in df.columns else "No Title",
            "Text": text[:100] + "..." if len(text) > 100 else text,
            "Sentiment Score": sentiment_score,
            "Market Impact": impact,
        }
        
        if 'Date' in df.columns:
            item["Date"] = row['Date']
        if 'Source' in df.columns:
            item["Source"] = row['Source']
        if 'Links' in df.columns:
            item["Link"] = row['Links']
        
        result.append(item)
    
    return pd.DataFrame(result)

# Streamlit App
st.set_page_config(page_title="Lithium Sentiment Dashboard", layout="wide")
st.title("🔋 Real-Time Lithium Sentiment Dashboard")

# Time period selection
time_period = st.sidebar.selectbox(
    "Select Time Period",
    options=[1, 2, 3, 5, 6],
    format_func=lambda x: f"{x} {'Month' if x == 1 else 'Months'}",
    index=4
)

try:
    with st.spinner("Connecting to database and inspecting structure..."):
        all_tables = list_tables()
        st.info(f"Available tables in database: {all_tables}")
    
    with st.spinner(f"Fetching and analyzing news from the last {time_period} month(s)..."):
        df = fetch_data_from_db(time_period)
        
        if not df.empty:
            analyzed_df = analyze_sentiment(df)
            st.subheader(f"📰 News Headlines & Market Impact (Last {time_period} month(s))")
            
            display_df = analyzed_df.copy()
            
            if 'Link' in display_df.columns:
                display_df['Title'] = display_df.apply(
                    lambda row: f'<a href="{row["Link"]}" target="_blank">{row["Title"]}</a>',
                    axis=1
                )
            
            display_cols = [col for col in ['Date', 'Title', 'Text', 'Sentiment Score', 'Market Impact', 'Source'] if col in display_df.columns]
            latest_display_df = display_df.sort_values(by='Date', ascending=False).head(8)
            latest_display_df = latest_display_df[display_cols].copy()
            st.markdown(latest_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            
            if 'Date' in analyzed_df.columns:
                st.subheader("📊 Sentiment Trend Over Time")

                trend_df = analyzed_df.copy()
                trend_df['Date'] = pd.to_datetime(trend_df['Date'], errors='coerce')
                trend_df = trend_df.dropna(subset=['Date']).set_index('Date')
    
                # Sidebar toggle for chart range
                trend_range = st.sidebar.radio(
                    "Select Sentiment Trend Range:",
                    options=["1 Week", "1 Month", "5 Months", "12 Months"],
                    index=3
                )
    
                # Convert selected range to timedelta
                range_map = {
                    "1 Week": 7,
                    "1 Month": 30,
                    "5 Months": 150,
                    "12 Months": 365
                }
                days_back = range_map[trend_range]
                trend_filtered = trend_df[trend_df.index >= (datetime.now() - timedelta(days=days_back))]

                # Line Chart
                daily_sentiment = trend_filtered.resample('D')['Sentiment Score'].mean().dropna()
                st.line_chart(daily_sentiment)
                # Fetch Lithium Prices
                lithium_price_df = fetch_lithium_prices()

                if not lithium_price_df.empty:
                    st.subheader("📈 Sentiment vs Lithium Price Trend")

                    # Prepare for merging
                    combined_df = daily_sentiment.reset_index().rename(columns={'Date': 'Dates'})
                    combined_df = pd.merge(combined_df, lithium_price_df, on='Dates', how='left')

                    combined_df = combined_df.dropna()

                    # Plotting with Altair (for dual Y-axis)
                    import altair as alt

                    base = alt.Chart(combined_df).encode(
                        x='Dates:T'
                    )

                    sentiment_line = base.mark_line(color='blue').encode(
                        y=alt.Y('Sentiment Score:Q', axis=alt.Axis(title='Sentiment Score'))
                    )

                    price_line = base.mark_line(color='green').encode(
                        y=alt.Y('Price:Q', axis=alt.Axis(title='Lithium Price'), scale=alt.Scale(zero=False))
                    )

                    chart = alt.layer(sentiment_line, price_line).resolve_scale(
                        y='independent'
                    ).properties(
                        width=800,
                        height=400
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Lithium Price data not available.")

                # Sentiment Distribution
                st.subheader("Sentiment Distribution")
                sentiment_counts = analyzed_df['Market Impact'].value_counts()
                st.bar_chart(sentiment_counts)

                # Average Sentiment
                avg_sentiment = analyzed_df['Sentiment Score'].mean()
                st.metric(
                    label="Average Sentiment Score",
                    value=f"{avg_sentiment:.3f}",
                    delta="Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
                )
        else:
            st.error("No data could be retrieved from the database.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.write("Exception details:", e)
