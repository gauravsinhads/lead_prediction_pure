import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("leads_prediction.csv", encoding='utf-8-sig')

    df.columns = df.columns.str.strip().str.upper()

    df.rename(columns={
        'MONTH_YEAR': 'month_year',
        'LEADS': 'Leads',
        'HIRED': 'Hired'
    }, inplace=True)

    df['month_year'] = pd.to_datetime(df['month_year'])

    df = df.groupby(['month_year','CAMPAIGN_SITE','BROADSOURCE'], as_index=False).agg({
        'Leads':'sum',
        'Hired':'sum'
    })

    df['conversion_rate'] = df['Hired'] / df['Leads']
    df['conversion_rate'] = df['conversion_rate'].replace([np.inf, -np.inf], 0).fillna(0)

    return df


df = load_data()

# -------------------------------
# TIME SETUP
# -------------------------------
current_month = df['month_year'].max()
prediction_month = current_month + pd.DateOffset(months=1)

train_end = current_month - pd.DateOffset(months=3)
train_df = df[df['month_year'] <= train_end]

# -------------------------------
# ARIMA MODEL
# -------------------------------
@st.cache_data
def run_arima(train_df):
    predictions = []

    for (site, source), group in train_df.groupby(['CAMPAIGN_SITE','BROADSOURCE']):
        ts = group.sort_values('month_year').set_index('month_year')['Leads']

        if len(ts) < 3:
            continue

        try:
            model = ARIMA(ts, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]

            predictions.append({
                'CAMPAIGN_SITE': site,
                'BROADSOURCE': source,
                'Predicted_Leads': max(float(forecast), 0)
            })
        except:
            continue

    return pd.DataFrame(predictions)


pred_df = run_arima(train_df)

# -------------------------------
# HISTORICAL METRICS
# -------------------------------
hist = df.groupby(['CAMPAIGN_SITE','BROADSOURCE']).agg({
    'Leads':'sum',
    'Hired':'sum'
}).reset_index()

hist['share_hired'] = hist['Hired'] / hist.groupby('CAMPAIGN_SITE')['Hired'].transform('sum')
hist['conversion_rate'] = hist['Hired'] / hist['Leads']
hist = hist.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------------------
# FINAL LEADS FUNCTION (PURE ARIMA)
# -------------------------------
def compute_final_leads(base, df, site=None):

    results = []

    for _, row in base.iterrows():

        source = row['BROADSOURCE']

        predicted = float(row.get('Predicted_Leads', 0))

        if np.isnan(predicted) or np.isinf(predicted):
            predicted = 0

        final = predicted  # PURE ARIMA

        if site:
            max_leads = df[
                (df['CAMPAIGN_SITE'] == site) &
                (df['BROADSOURCE'] == source)
            ]['Leads'].max()
        else:
            max_leads = df[df['BROADSOURCE'] == source]['Leads'].max()

        if pd.isna(max_leads):
            limit = final
        else:
            limit = 1.5 * float(max_leads)

        capped = min(final, limit)
        excess = final - capped

        results.append({
            'BROADSOURCE': source,
            'Lead Count Required': capped,
            'excess': excess
        })

    final_df = pd.DataFrame(results)

    if 'Social Media' in final_df['BROADSOURCE'].values:
        excess_total = final_df['excess'].sum()
        final_df.loc[
            final_df['BROADSOURCE'] == 'Social Media',
            'Lead Count Required'
        ] += excess_total

    return final_df[['BROADSOURCE','Lead Count Required']]

# -------------------------------
# ROLLING ACCURACY (PURE ARIMA)
# -------------------------------
rolling_results = []

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[df['month_year'] < test_month]
    test_temp = df[df['month_year'] == test_month]

    pred_temp = run_arima(train_temp)

    base = test_temp.copy()

    base = base.merge(pred_temp, on=['CAMPAIGN_SITE','BROADSOURCE'], how='left')
    base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

    base['final_leads'] = base['Predicted_Leads']  # PURE ARIMA

    actual_total = base['Leads'].sum()
    predicted_total = base['final_leads'].sum()

    rmse = abs(actual_total - predicted_total)

    if actual_total != 0:
        mape = rmse / actual_total
    else:
        mape = 0

    rolling_results.append({
        'Month': test_month.strftime('%Y-%m'),
        'Actual Leads': round(actual_total, 2),
        'Predicted Leads (Final)': round(predicted_total, 2),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape * 100, 2)
    })

rolling_accuracy_df = pd.DataFrame(rolling_results)

# -------------------------------
# SITE-LEVEL ACCURACY (PURE ARIMA)
# -------------------------------
site_level_results = []

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[df['month_year'] < test_month]
    test_temp = df[df['month_year'] == test_month]

    pred_temp = run_arima(train_temp)

    base = test_temp.copy()

    base = base.merge(pred_temp, on=['CAMPAIGN_SITE','BROADSOURCE'], how='left')
    base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

    base['final_leads'] = base['Predicted_Leads']

    for site_name, grp in base.groupby('CAMPAIGN_SITE'):

        actual_total = grp['Leads'].sum()
        predicted_total = grp['final_leads'].sum()

        rmse = abs(actual_total - predicted_total)

        if actual_total != 0:
            mape = rmse / actual_total
        else:
            mape = 0

        site_level_results.append({
            'Month': test_month.strftime('%Y-%m'),
            'CAMPAIGN_SITE': site_name,
            'Actual Leads': round(actual_total, 2),
            'Predicted Leads (Final)': round(predicted_total, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape * 100, 2)
        })

site_level_accuracy_df = pd.DataFrame(site_level_results)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📊 Lead Prediction Calculator (Pure ARIMA)")

st.info(f"📅 Prediction Month: {prediction_month.strftime('%Y-%m')}")

st.sidebar.header("📉 Accuracy (Final Output Based)")
st.sidebar.dataframe(rolling_accuracy_df)

st.sidebar.subheader("📍 Site-Level Accuracy")
st.sidebar.dataframe(site_level_accuracy_df)

site_options = ["All Sites"] + sorted(df['CAMPAIGN_SITE'].unique())
site = st.selectbox("Select Campaign Site", site_options)

target_hired = st.number_input("Enter Target HIRED", min_value=0, step=1)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):

    if site == "All Sites":

        base = pred_df.groupby('BROADSOURCE')['Predicted_Leads'].sum().reset_index()
        base.rename(columns={'Predicted_Leads': 'Lead Count Required'}, inplace=True)
        base['CAMPAIGN_SITE'] = "All Sites"

    else:
        base = pred_df[pred_df['CAMPAIGN_SITE'] == site][['BROADSOURCE','Predicted_Leads']]
        base = base.rename(columns={'Predicted_Leads': 'Lead Count Required'})
        base['CAMPAIGN_SITE'] = site

    base['Lead Count Required'] = base['Lead Count Required'].round().astype(int)

    final_output = base[['CAMPAIGN_SITE','BROADSOURCE','Lead Count Required']]

    st.subheader("📈 Final Lead Plan")
    st.dataframe(final_output)

    st.bar_chart(final_output.set_index('BROADSOURCE')['Lead Count Required'])
