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
# TRAIN MODEL
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
# PREDICTION
# -------------------------------
def generate_predictions(model, df, prediction_month):

    latest = df.sort_values('month_year').groupby(
        ['CAMPAIGN_SITE','BROADSOURCE']
    ).tail(1)

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
# HISTORICAL METRICS (UNCHANGED)
# -------------------------------
hist = df.groupby(['CAMPAIGN_SITE','BROADSOURCE']).agg({
    'Leads':'sum',
    'Hired':'sum'
}).reset_index()

hist['share_hired'] = hist['Hired'] / hist.groupby('CAMPAIGN_SITE')['Hired'].transform('sum')
hist['conversion_rate'] = hist['Hired'] / hist['Leads']
hist = hist.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------------------
# FINAL LEADS FUNCTION (PURE ML)
# -------------------------------
def compute_final_leads(base, df, site=None):

    results = []

    for _, row in base.iterrows():

        source = row['BROADSOURCE']
        predicted = float(row.get('Predicted_Leads', 0))

        if np.isnan(predicted) or np.isinf(predicted):
            predicted = 0

        final = predicted  # ✅ CHANGED HERE

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

for i in range(3, 0, -1):

    test_month = current_month - pd.DateOffset(months=i)

    train_temp = df[df['month_year'] < test_month]
    test_temp = df[df['month_year'] == test_month]

    model_temp = train_xgboost(train_temp)
    pred_temp = generate_predictions(model_temp, df, test_month)

    base = test_temp.copy()

    base = base.merge(pred_temp, on=['CAMPAIGN_SITE','BROADSOURCE'], how='left')
    base['Predicted_Leads'] = base['Predicted_Leads'].fillna(0)

    base['final_leads'] = base['Predicted_Leads']  # ✅ CHANGED

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
# SITE-LEVEL ACCURACY (UPDATED)
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

    base['final_leads'] = base['Predicted_Leads']  # ✅ CHANGED

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
# STREAMLIT UI (UNCHANGED)
# -------------------------------
st.title("📊 Lead Prediction Calculator (Pure XGBoost Output)")

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
        xgb_agg = pred_df.groupby('BROADSOURCE')['Predicted_Leads'].sum().reset_index()
        output = xgb_agg.copy()
        output['CAMPAIGN_SITE'] = "All Sites"
        output.rename(columns={'Predicted_Leads':'Lead Count Required'}, inplace=True)

    else:
        xgb_site = pred_df[pred_df['CAMPAIGN_SITE'] == site]
        output = xgb_site.copy()
        output.rename(columns={'Predicted_Leads':'Lead Count Required'}, inplace=True)

    output['Lead Count Required'] = output['Lead Count Required'].round().astype(int)

    final_output = output[['CAMPAIGN_SITE','BROADSOURCE','Lead Count Required']]

    st.subheader("📈 Final Lead Plan (Pure ML)")
    st.dataframe(final_output)

    st.bar_chart(final_output.set_index('BROADSOURCE')['Lead Count Required'])
