import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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
# FEATURE ENGINEERING
# -------------------------------
def create_features(data):
    data = data.copy()
    data['month_num'] = data['month_year'].dt.month
    data['year'] = data['month_year'].dt.year
    data['lag_1'] = data.groupby(['CAMPAIGN_SITE','BROADSOURCE'])['Leads'].shift(1)
    data['lag_2'] = data.groupby(['CAMPAIGN_SITE','BROADSOURCE'])['Leads'].shift(2)
    return data

# -------------------------------
# FAST AUTOML
# -------------------------------
@st.cache_resource
def train_models(train_df):

    df_feat = create_features(train_df).dropna()

    models_dict = {}

    for (site, source), group in df_feat.groupby(['CAMPAIGN_SITE','BROADSOURCE']):

        if len(group) < 5:
            continue

        X = group[['month_num','year','lag_1','lag_2']]
        y = group['Leads']

        models = {
            "lr": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=50, random_state=42)
        }

        best_model = None
        best_rmse = float('inf')

        for model in models.values():
            try:
                model.fit(X, y)
                preds = model.predict(X)
                rmse = np.sqrt(mean_squared_error(y, preds))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
            except:
                continue

        if best_model:
            models_dict[(site, source)] = best_model

    return models_dict

# -------------------------------
# PREDICTION
# -------------------------------
def run_automl(train_df):

    models_dict = train_models(train_df)

    df_feat = create_features(train_df).dropna()

    predictions = []

    for (site, source), model in models_dict.items():

        group = df_feat[
            (df_feat['CAMPAIGN_SITE'] == site) &
            (df_feat['BROADSOURCE'] == source)
        ]

        try:
            last_row = group.sort_values('month_year').iloc[-1]

            future = pd.DataFrame({
                'month_num': [(last_row['month_num'] % 12) + 1],
                'year': [last_row['year'] + (1 if last_row['month_num'] == 12 else 0)],
                'lag_1': [last_row['Leads']],
                'lag_2': [group.iloc[-2]['Leads']]
            })

            forecast = model.predict(future)[0]

            predictions.append({
                'CAMPAIGN_SITE': site,
                'BROADSOURCE': source,
                'Predicted_Leads': max(float(forecast), 0)
            })

        except:
            continue

    return pd.DataFrame(predictions)

pred_df = run_automl(train_df)

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
# FINAL LEADS FUNCTION (UPDATED)
# -------------------------------
def compute_final_leads(base, df, site=None):

    results = []

    for _, row in base.iterrows():

        source = row['BROADSOURCE']
        predicted = float(row.get('Predicted_Leads', 0))

        predicted = 0 if np.isnan(predicted) or np.isinf(predicted) else predicted

        final = predicted  # 🔥 PURE AUTOML

        if site:
            max_leads = df[
                (df['CAMPAIGN_SITE'] == site) &
                (df['BROADSOURCE'] == source)
            ]['Leads'].max()
        else:
            max_leads = df[df['BROADSOURCE'] == source]['Leads'].max()

        limit = final if pd.isna(max_leads) else 1.5 * float(max_leads)

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
# ROLLING ACCURACY (UPDATED)
# -------------------------------
rolling_results = []
site_level_results = []

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    test_temp = df[df['month_year'] == test_month]
    base = test_temp.copy()

    base = base.merge(pred_df, on=['CAMPAIGN_SITE','BROADSOURCE'], how='left')
    base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

    base['final_leads'] = base['Predicted_Leads']  # 🔥 PURE AUTOML

    actual_total = base['Leads'].sum()
    predicted_total = base['final_leads'].sum()

    rmse = abs(actual_total - predicted_total)
    mape = rmse / actual_total if actual_total != 0 else 0

    rolling_results.append({
        'Month': test_month.strftime('%Y-%m'),
        'Actual Leads': round(actual_total, 2),
        'Predicted Leads (Final)': round(predicted_total, 2),
        'RMSE': round(rmse, 2),
        'MAPE (%)': round(mape * 100, 2)
    })

    for site_name, grp in base.groupby('CAMPAIGN_SITE'):

        actual_total = grp['Leads'].sum()
        predicted_total = grp['final_leads'].sum()

        rmse = abs(actual_total - predicted_total)
        mape = rmse / actual_total if actual_total != 0 else 0

        site_level_results.append({
            'Month': test_month.strftime('%Y-%m'),
            'CAMPAIGN_SITE': site_name,
            'Actual Leads': round(actual_total, 2),
            'Predicted Leads (Final)': round(predicted_total, 2),
            'RMSE': round(rmse, 2),
            'MAPE (%)': round(mape * 100, 2)
        })

rolling_accuracy_df = pd.DataFrame(rolling_results)
site_level_accuracy_df = pd.DataFrame(site_level_results)

# -------------------------------
# UI
# -------------------------------
st.title("📊 Lead Prediction Calculator (Pure AutoML)")

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

        output = compute_final_leads(base, df, site=None)
        output['CAMPAIGN_SITE'] = "All Sites"

    else:
        base = pred_df[pred_df['CAMPAIGN_SITE'] == site]

        output = compute_final_leads(base, df, site=site)
        output['CAMPAIGN_SITE'] = site

    output['Lead Count Required'] = output['Lead Count Required'].round().astype(int)

    final_output = output[['CAMPAIGN_SITE','BROADSOURCE','Lead Count Required']]

    st.subheader("📈 Final Lead Plan")
    st.dataframe(final_output)

    st.bar_chart(final_output.set_index('BROADSOURCE')['Lead Count Required'])
