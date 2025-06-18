import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# --- Допоміжні функції (визначені поза кешованими функціями) ---

def handle_outliers(data, method="clip"):
    """Correctly handles outliers for all numeric columns."""
    data_copy = data.copy()
    for column in data_copy.select_dtypes(include="number").columns:
        Q1 = data_copy[column].quantile(0.25)
        Q3 = data_copy[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # FIX: This block is now inside the loop
        if method == "clip":
            data_copy[column] = data_copy[column].clip(lower, upper)
        elif method == "replace":
            median = data_copy[column].median()
            data_copy[column] = data_copy[column].apply(
                lambda x: median if x < lower or x > upper else x
            )
    return data_copy

# --- Кешовані функції для продуктивності ---

@st.cache_data
def load_and_prepare_data():
    """Loads and performs all pre-processing steps on the data."""
    df = pd.read_csv("Supermarket_customers.csv", sep="\t")
    
    # Age calculation
    current_year = pd.to_datetime("today").year
    birth_median = df["Year_Birth"].median()
    df.loc[df["Year_Birth"] < 1940, "Year_Birth"] = birth_median
    df["Age"] = current_year - df["Year_Birth"]
    df = df.drop("Year_Birth", axis=1)
    
    df_processed = df.copy()
    
    # Feature Engineering & Cleaning
    df_processed = handle_outliers(df_processed)
    income_median = df_processed["Income"].median()
    df_processed["Income"] = df_processed["Income"].fillna(income_median)
    df_processed["Marital_Status"] = df_processed["Marital_Status"].replace(["Absurd", "YOLO"], "Married")
    df_processed["Total_Children"] = df_processed["Kidhome"] + df_processed["Teenhome"]
    df_processed["Dt_Customer"] = pd.to_datetime(df_processed["Dt_Customer"], format="%d-%m-%Y", errors="coerce")
    df_processed["Customer_Since"] = (pd.to_datetime("today") - df_processed["Dt_Customer"]).dt.days
    mnt_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    df_processed["MntTotal"] = df_processed[mnt_cols].sum(axis=1)
    purchase_cols = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df_processed["TotalPurchases"] = df_processed[purchase_cols].sum(axis=1)
    cols_to_drop = ["ID", "Z_CostContact", "Z_Revenue", "Dt_Customer", "Kidhome", "Teenhome"]
    df_processed = df_processed.drop(columns=cols_to_drop)

    # Prepare for modeling
    X = pd.get_dummies(df_processed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df_processed, X_scaled

@st.cache_resource
def train_model(X_scaled):
    """Trains the clustering model and performs PCA."""
    final_agg = AgglomerativeClustering(n_clusters=2)
    clusters = final_agg.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return clusters, X_pca

# --- Основна частина додатку ---

st.set_page_config(layout="wide")
st.title("Segmentation Analysis of Supermarket Customers")

# 1. Завантаження, підготовка та моделювання
df_processed, X_scaled = load_and_prepare_data()
clusters, X_pca = train_model(X_scaled)

# Додавання результатів кластеризації
df_processed["Cluster"] = clusters
pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["Cluster"] = clusters

# 2. FIX: Динамічне визначення та перейменування кластерів
income_means = df_processed.groupby('Cluster')['Income'].mean()
sorted_clusters = income_means.sort_values().index
cluster_names_map = {
    sorted_clusters[0]: "Economical Families",
    sorted_clusters[-1]: "Wealthy Gourmets"
}
df_processed['Cluster_Name'] = df_processed['Cluster'].map(cluster_names_map)
pca_df['Cluster_Name'] = pca_df['Cluster'].map(cluster_names_map)
cluster_options = list(cluster_names_map.values())

# --- БІЧНА ПАНЕЛЬ (SIDEBAR) ---
st.sidebar.header("Explore Clusters")
selected_cluster_name = st.sidebar.selectbox(
    "Select a cluster to see its data:",
    options=cluster_options
)

st.sidebar.header("Filters for Visualisation")
income_range = st.sidebar.slider(
    "Filter by Income:",
    min_value=int(df_processed['Income'].min()),
    max_value=int(df_processed['Income'].max()),
    value=(int(df_processed['Income'].min()), int(df_processed['Income'].max()))
)

age_range = st.sidebar.slider(
    "Filter by Age:",
    min_value=int(df_processed['Age'].min()),
    max_value=int(df_processed['Age'].max()),
    value=(int(df_processed['Age'].min()), int(df_processed['Age'].max()))
)          
marital_options = df_processed['Marital_Status'].unique().tolist()
selected_marital_statuses = st.sidebar.multiselect(
    "Filter by Marital Status:",
    options=marital_options,
    default=marital_options  # За замовчуванням вибрані всі статуси
)
children_options = sorted(df_processed['Total_Children'].unique().tolist())
selected_children_counts = st.sidebar.multiselect(
    "Filter by Total Children:",
    options=children_options,
    default=children_options # За замовчуванням вибрані всі варіанти
)
  
numCatalogPurchases_range = st.sidebar.slider(
"Filter by NumCatalogPurchases:",
    min_value=int(df_processed['NumCatalogPurchases'].min()),
    max_value=int(df_processed['NumCatalogPurchases'].max()),
    value=(int(df_processed['NumCatalogPurchases'].min()), int(df_processed['NumCatalogPurchases'].max()))
)
mntMeatProducts_range = st.sidebar.slider(
"Filter by 'MntMeatProducts:",
    min_value=int(df_processed['MntMeatProducts'].min()),
    max_value=int(df_processed['MntMeatProducts'].max()),
    value=(int(df_processed['MntMeatProducts'].min()), int(df_processed['MntMeatProducts'].max()))
)
# --- ОСНОВНА СТОРІНКА ---

# 3. Візуалізація та аналіз
tab1, tab2, tab3 = st.tabs(["📊 Cluster Overview", "🔬 Detailed Analysis", "↔️ Cluster Comparison"])

with tab1:
    st.header("Customer Cluster Visualisation")
    st.write("Clusters visualised using Principal Component Analysis (PCA).")

    # Фільтруємо дані для графіка на основі слайдера
    #filtered_df = df_processed[(df_processed['Income'] >= income_range[0]) & (df_processed['Income'] <= income_range[1])]
    #filtered_pca_df = pca_df.loc[filtered_df.index]
    

    filtered_df = df_processed[
        (df_processed['Income'].between(income_range[0], income_range[1])) &
        (df_processed['Age'].between(age_range[0], age_range[1])) &
        (df_processed['Marital_Status'].isin(selected_marital_statuses)) &
        (df_processed['Total_Children'].isin(selected_children_counts)) &
        (df_processed['NumCatalogPurchases'].between(numCatalogPurchases_range[0], numCatalogPurchases_range[1])) &
        (df_processed['MntMeatProducts'].between(mntMeatProducts_range[0], mntMeatProducts_range[1]))
    ]
        #(df_processed['Marital_Status'].isin(selected_marital_statuses)) &
        #(df_processed['Total_Children'].isin(selected_children_counts))
    #]
    if not filtered_df.empty:
        filtered_pca_df = pca_df.loc[filtered_df.index]
        
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            x="PC1", y="PC2", hue="Cluster_Name", data=filtered_pca_df,
            palette="viridis", s=100, alpha=0.7, ax=ax1
        )
        ax1.set_title("Customer Clusters (PCA)")
        st.pyplot(fig1)
    else:
        # Якщо після фільтрації не залишилося клієнтів
        st.warning("No customers match the selected filter criteria. Please adjust the filters.")




   # fig1, ax1 = plt.subplots(figsize=(10, 7))
    #sns.scatterplot(
       # x="PC1", y="PC2", hue="Cluster_Name", data=filtered_pca_df,
        #palette="viridis", s=100, alpha=0.7, ax=ax1
   # )
    #ax1.set_title("Customer Clusters (PCA)")
    #st.pyplot(fig1)

    st.header("Description of Customer Segments")
    st.subheader("Economical Families")
    st.markdown("- **Characteristics:** Lower income, often have children, responsive to deals and discounts.")
    st.markdown("- **Marketing Strategy:** Offer coupons, promotions on family-sized products, loyalty programs.")
    st.subheader("Wealthy Gourmets")
    st.markdown("- **Characteristics:** High income, fewer children, actively buy wine and premium meat/fish products.")
    st.markdown("- **Marketing Strategy:** Personalised offers for expensive goods, wine tastings, premium service.")

with tab2:
    st.header(f"Detailed Analysis of Cluster: {selected_cluster_name}")
    
    cluster_data = df_processed[df_processed['Cluster_Name'] == selected_cluster_name]
    st.write("First 10 customers from this segment:")
    st.dataframe(cluster_data.head(10))

    st.write("Key characteristic distributions for this cluster:")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.histplot(cluster_data['Income'], bins=20, kde=True, ax=ax).set_title("Income Distribution")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(cluster_data['MntTotal'], bins=20, kde=True, ax=ax).set_title("Total Products Spending")
        st.pyplot(fig)
    with col3:
        fig, ax = plt.subplots()
        sns.histplot(cluster_data['TotalPurchases'], bins=20, kde=True, ax=ax).set_title("Total Purchases")
        st.pyplot(fig)

with tab3:
    st.header("Cluster Profile Comparison")
    st.write("Compare the average values of characteristics across clusters.")
    
    # Розраховуємо середні значення, групуючи за назвами
    df_processed = df_processed.drop ([ 'AcceptedCmp3','AcceptedCmp4','AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2','Complain', 'Response'],axis=1)
    cluster_means = df_processed.groupby("Cluster_Name").mean(numeric_only=True)

    # Масштабуємо для коректної візуалізації
    scaler = MinMaxScaler()
    scaled_means_data = scaler.fit_transform(cluster_means)
    scaled_cluster_means_df = pd.DataFrame(
        data=scaled_means_data,
        index=cluster_means.index,
        columns=cluster_means.columns
    )
   
    st.subheader("Comparison of Scaled Profiles")
    fig, ax = plt.subplots(figsize=(16, 8))
    scaled_cluster_means_df.T.plot(kind='bar', ax=ax)
    ax.set_title("Relative Average Values of Characteristics by Cluster")
    ax.set_ylabel("Scaled Value (from 0 to 1)")
    ax.set_xlabel("Characteristics")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Comparison of Mean Values (Raw Data)")
    st.dataframe(cluster_means.T.style.format("{:.2f}"))