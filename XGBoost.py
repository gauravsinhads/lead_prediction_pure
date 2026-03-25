import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

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
# FEATURE ENGINEERING
# -------------------------------
def create_features(df):
    df = df.copy()

    df['month'] = df['month_year'].dt.month
    df['year'] = df['month_year'].dt.year

    df['site_id'] = df['CAMPAIGN_SITE'].astype('category').cat.codes
    df['source_id'] = df['BROADSOURCE'].astype('category').cat.codes

    df = df.sort_values('month_year')

    df['lag_1'] = df.groupby(['CAMPAIGN_SITE','BROADSOURCE'])['Leads'].shift(1)
    df['lag_2'] = df.groupby(['CAMPAIGN_SITE','BROADSOURCE'])['Leads'].shift(2)

    df = df.fillna(0)

    return df

df = create_features(df)

# -------------------------------
# TIME SETUP
# -------------------------------
current_month = df['month_year'].max()
prediction_month = current_month + pd.DateOffset(months=1)

train_end = current_month - pd.DateOffset(months=3)
train_df = df[df['month_year'] <= train_end]

# -------------------------------
# TRAIN MODEL (CACHE SAFE)
# -------------------------------
@st.cache_resource
def train_xgboost(train_df):

    features = ['month','year','lag_1','lag_2','site_id','source_id']

    X = train_df[features]
    y = train_df['Leads']

    model = XGBRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model

# -------------------------------
# PREDICTION (NO CACHE → FIX)
# -------------------------------
def generate_predictions(model, df, prediction_month):

    latest = df.sort_values('month_year').groupby(
        ['CAMPAIGN_SITE','BROADSOURCE']
    ).tail(1)

    if latest.empty:
        return pd.DataFrame(columns=['CAMPAIGN_SITE','BROADSOURCE','Predicted_Leads'])

    pred_df = latest.copy()

    pred_df['month'] = prediction_month.month
    pred_df['year'] = prediction_month.year

    features = ['month','year','lag_1','lag_2','site_id','source_id']

    preds = model.predict(pred_df[features])

    pred_df['Predicted_Leads'] = np.maximum(preds, 0)

    return pred_df[['CAMPAIGN_SITE','BROADSOURCE','Predicted_Leads']]

# Train once
model = train_xgboost(train_df)
pred_df = generate_predictions(model, df, prediction_month)

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
# FINAL LEADS FUNCTION
# -------------------------------
def compute_final_leads(base, df, site=None):

    results = []

    for _, row in base.iterrows():

        source = row['BROADSOURCE']

        required = float(row.get('required_leads', 0))
        predicted = float(row.get('Predicted_Leads', 0))

        required = 0 if np.isnan(required) or np.isinf(required) else required
        predicted = 0 if np.isnan(predicted) or np.isinf(predicted) else predicted

        final = max(required, predicted)

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
# ROLLING ACCURACY
# -------------------------------
rolling_results = []

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[df['month_year'] < test_month]
    test_temp = df[df['month_year'] == test_month]

    model_temp = train_xgboost(train_temp)
    pred_temp = generate_predictions(model_temp, df, test_month)

    base = test_temp.copy()

    base = base.merge(pred_temp, on=['CAMPAIGN_SITE','BROADSOURCE'], how='left')
    base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

    base['required_leads'] = base['Hired'] / base['conversion_rate']
    base['required_leads'] = base['required_leads'].replace([np.inf, -np.inf], 0).fillna(0)

    base['final_leads'] = base[['required_leads','Predicted_Leads']].max(axis=1)
    base['final_leads'] = base['final_leads'].replace([np.inf, -np.inf], 0).fillna(0)

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

rolling_accuracy_df = pd.DataFrame(rolling_results)

# -------------------------------
# SITE-LEVEL ACCURACY
# -------------------------------
site_level_results = []

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[df['month_year'] < test_month]
    test_temp = df[df['month_year'] == test_month]

    model_temp = train_xgboost(train_temp)
    pred_temp = generate_predictions(model_temp, df, test_month)

    base = test_temp.copy()

    base = base.merge(pred_temp, on=['CAMPAIGN_SITE','BROADSOURCE'], how='left')
    base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

    base['required_leads'] = base['Hired'] / base['conversion_rate']
    base['required_leads'] = base['required_leads'].replace([np.inf, -np.inf], 0).fillna(0)

    base['final_leads'] = base[['required_leads','Predicted_Leads']].max(axis=1)
    base['final_leads'] = base['final_leads'].replace([np.inf, -np.inf], 0).fillna(0)

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

site_level_accuracy_df = pd.DataFrame(site_level_results)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("📊 Lead Prediction Calculator (XGBoost - Stable & Fast)")

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

        base = df.groupby('BROADSOURCE').agg({
            'Leads':'sum',
            'Hired':'sum'
        }).reset_index()

        base['share_hired'] = base['Hired'] / base['Hired'].sum()
        base['conversion_rate'] = base['Hired'] / base['Leads']

        base['target_hired'] = base['share_hired'] * target_hired
        base['required_leads'] = base['target_hired'] / base['conversion_rate']
        base['required_leads'] = base['required_leads'].replace([np.inf, -np.inf], 0).fillna(0)

        xgb_agg = pred_df.groupby('BROADSOURCE')['Predicted_Leads'].sum().reset_index()

        base = base.merge(xgb_agg, on='BROADSOURCE', how='left')
        base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

        output = compute_final_leads(base, df, site=None)
        output['CAMPAIGN_SITE'] = "All Sites"

    else:
        base = hist[hist['CAMPAIGN_SITE'] == site].copy()

        base['target_hired'] = base['share_hired'] * target_hired
        base['required_leads'] = base['target_hired'] / base['conversion_rate']
        base['required_leads'] = base['required_leads'].replace([np.inf, -np.inf], 0).fillna(0)

        xgb_site = pred_df[pred_df['CAMPAIGN_SITE'] == site]

        base = base.merge(xgb_site[['BROADSOURCE','Predicted_Leads']], on='BROADSOURCE', how='left')
        base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

        output = compute_final_leads(base, df, site=site)
        output['CAMPAIGN_SITE'] = site

    output['Lead Count Required'] = output['Lead Count Required'].round().astype(int)

    final_output = output[['CAMPAIGN_SITE','BROADSOURCE','Lead Count Required']]

    st.subheader("📈 Final Lead Plan")
    st.dataframe(final_output)

    st.bar_chart(final_output.set_index('BROADSOURCE')['Lead Count Required'])
