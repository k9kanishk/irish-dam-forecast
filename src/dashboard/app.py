
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title='Irish DAM Forecast', layout='wide')

@st.cache_data
def load():
    return pd.read_parquet('data/processed/train.parquet')


df = load()

y = df.pop('target')
mdl = __import__('src.models.xgb_model', fromlist=['make_model']).make_model()
mdl.fit(df, y)

st.title('Irish Day‑Ahead Power Price Forecast (SEMOpx)')

from datetime import datetime
from zoneinfo import ZoneInfo
now = datetime.now(ZoneInfo('Europe/Dublin')).date()

date = st.date_input('Choose a date to forecast', now)
mask = (df.index.date == pd.to_datetime(date).date())
X_day = df.loc[mask]
if len(X_day):
    preds = mdl.predict(X_day)
    out = pd.DataFrame({'ts': X_day.index, 'forecast_eur_mwh': preds}).set_index('ts')
    st.subheader('Hourly forecast')
    st.dataframe(out)
    st.plotly_chart(px.line(out, y='forecast_eur_mwh'), use_container_width=True)
else:
    st.info('No features for this day yet. Run data fetch scripts for the target date range.')
