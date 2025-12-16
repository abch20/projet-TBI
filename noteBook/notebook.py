# Advanced Database Analysis
# Schema: Dim_Client_final, Fait_Commande, Dim_Temps1, Dim_Employees_final

import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

# Load your data (adjust paths and connection as needed)
# Option 1: From CSV files
df_clients = pd.read_csv('Data\Dim_Client_final.csv')
df_commandes = pd.read_csv('Data\Fait Commande.csv')
df_temps = pd.read_csv('Data\Dim_Temps1.csv')
df_employees = pd.read_csv('Data\Dim_Employees_final.csv')

# Option 2: From database
# import sqlalchemy
# engine = sqlalchemy.create_engine('your_connection_string')
# df_clients = pd.read_sql_table('Dim_Client_final', engine)
# df_commandes = pd.read_sql_table('Fait_Commande', engine)
# df_temps = pd.read_sql_table('Dim_Temps1', engine)
# df_employees = pd.read_sql_table('Dim_Employees_final', engine)

# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================

def explore_data(df, name):
    """Comprehensive data exploration"""
    print(f"\n{'='*60}")
    print(f"EXPLORING: {name}")
    print(f"{'='*60}")
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print(f"\nFirst 5 rows:\n{df.head()}")

# Uncomment to explore each table
# explore_data(df_clients, "Dim_Client_final")
# explore_data(df_commandes, "Fait_Commande")
# explore_data(df_temps, "Dim_Temps1")
# explore_data(df_employees, "Dim_Employees_final")

# ============================================================================
# 3. DATA MERGING & ENRICHMENT
# ============================================================================

def create_master_dataset(df_commandes, df_clients, df_employees, df_temps):
    """Create a comprehensive dataset by joining all tables"""
    
    # Join commandes with clients
    df_master = df_commandes.merge(
        df_clients,
        left_on='id_seqClient',
        right_on='id_seqClient',
        how='left',
        suffixes=('', '_client')
    )
    
    # Join with employees
    df_master = df_master.merge(
        df_employees,
        left_on='id_seq_Employee',
        right_on='id_seq_Employee',
        how='left',
        suffixes=('', '_employee')
    )
    
    # Join with time dimension
    df_master = df_master.merge(
        df_temps,
        left_on='id_temps',
        right_on='id_temps',
        how='left',
        suffixes=('', '_temps')
    )
    
    return df_master

df_master = create_master_dataset(df_commandes, df_clients, df_employees, df_temps)
print(f"Master dataset shape: {df_master.shape}")

# ============================================================================
# 4. BUSINESS METRICS ANALYSIS
# ============================================================================

def calculate_kpis(df_master):
    """Calculate key performance indicators"""
    
    kpis = {
        'total_orders': len(df_master),
        'total_delivered': df_master['Nbr_Commande_Livree'].sum(),
        'total_undelivered': df_master['Nbr_Commande_Non_Livree'].sum(),
        'delivery_rate': (df_master['Nbr_Commande_Livree'].sum() / 
                         (df_master['Nbr_Commande_Livree'].sum() + 
                          df_master['Nbr_Commande_Non_Livree'].sum()) * 100),
        'unique_customers': df_master['Customer ID'].nunique(),
        'unique_employees': df_master['Employee ID'].nunique(),
    }
    
    return pd.DataFrame([kpis])

kpis = calculate_kpis(df_master)
print("\nKEY PERFORMANCE INDICATORS:")
print(kpis.T)

# ============================================================================
# 5. CUSTOMER ANALYSIS
# ============================================================================

def customer_analysis(df_master):
    """Analyze customer behavior and segmentation"""
    
    # Customer order frequency
    customer_orders = df_master.groupby('Customer ID').agg({
        'id_seq_fait': 'count',
        'Nbr_Commande_Livree': 'sum',
        'Nbr_Commande_Non_Livree': 'sum',
        'Country': 'first',
        'City': 'first'
    }).reset_index()
    
    customer_orders.columns = ['Customer_ID', 'Total_Orders', 'Delivered', 
                               'Undelivered', 'Country', 'City']
    
    # Calculate delivery success rate per customer
    customer_orders['Delivery_Rate'] = (
        customer_orders['Delivered'] / 
        (customer_orders['Delivered'] + customer_orders['Undelivered']) * 100
    )
    
    # Customer segmentation by order volume
    customer_orders['Segment'] = pd.cut(
        customer_orders['Total_Orders'],
        bins=[0, 1, 5, 10, float('inf')],
        labels=['One-time', 'Occasional', 'Regular', 'VIP']
    )
    
    return customer_orders

customer_stats = customer_analysis(df_master)
print("\nCUSTOMER SEGMENTS:")
print(customer_stats['Segment'].value_counts())

# ============================================================================
# 6. EMPLOYEE PERFORMANCE ANALYSIS
# ============================================================================

def employee_performance(df_master):
    """Analyze employee performance metrics"""
    
    employee_stats = df_master.groupby(['Employee ID', 'Nom', 'Prenom']).agg({
        'id_seq_fait': 'count',
        'Nbr_Commande_Livree': 'sum',
        'Nbr_Commande_Non_Livree': 'sum',
        'Region': 'first',
        'Territory': 'first'
    }).reset_index()
    
    employee_stats.columns = ['Employee_ID', 'Last_Name', 'First_Name', 
                              'Total_Orders', 'Delivered', 'Undelivered',
                              'Region', 'Territory']
    
    # Calculate performance metrics
    employee_stats['Delivery_Rate'] = (
        employee_stats['Delivered'] / 
        (employee_stats['Delivered'] + employee_stats['Undelivered']) * 100
    )
    
    employee_stats['Efficiency_Score'] = (
        employee_stats['Delivered'] / employee_stats['Total_Orders'] * 100
    )
    
    # Rank employees
    employee_stats['Rank'] = employee_stats['Delivery_Rate'].rank(
        ascending=False, method='dense'
    )
    
    return employee_stats.sort_values('Delivery_Rate', ascending=False)

employee_perf = employee_performance(df_master)
print("\nTOP 10 EMPLOYEES BY DELIVERY RATE:")
print(employee_perf.head(10))

# ============================================================================
# 7. TEMPORAL ANALYSIS
# ============================================================================

def temporal_analysis(df_master):
    """Analyze trends over time"""
    
    # Assuming df_temps has date-related columns (Annee, mois_annee, etc.)
    temporal = df_master.groupby(['Annee', 'mois_annee']).agg({
        'id_seq_fait': 'count',
        'Nbr_Commande_Livree': 'sum',
        'Nbr_Commande_Non_Livree': 'sum'
    }).reset_index()
    
    temporal.columns = ['Year', 'Month', 'Total_Orders', 
                        'Delivered', 'Undelivered']
    
    temporal['Delivery_Rate'] = (
        temporal['Delivered'] / 
        (temporal['Delivered'] + temporal['Undelivered']) * 100
    )
    
    return temporal

temporal_trends = temporal_analysis(df_master)
print("\nTEMPORAL TRENDS:")
print(temporal_trends.tail(12))

# ============================================================================
# 8. GEOGRAPHICAL ANALYSIS
# ============================================================================

def geographical_analysis(df_master):
    """Analyze performance by geography"""
    
    # By Country
    country_stats = df_master.groupby('Country').agg({
        'id_seq_fait': 'count',
        'Nbr_Commande_Livree': 'sum',
        'Nbr_Commande_Non_Livree': 'sum',
        'Customer ID': 'nunique'
    }).reset_index()
    
    country_stats.columns = ['Country', 'Total_Orders', 'Delivered', 
                             'Undelivered', 'Unique_Customers']
    
    country_stats['Delivery_Rate'] = (
        country_stats['Delivered'] / 
        (country_stats['Delivered'] + country_stats['Undelivered']) * 100
    )
    
    return country_stats.sort_values('Total_Orders', ascending=False)

geo_stats = geographical_analysis(df_master)
print("\nGEOGRAPHICAL PERFORMANCE:")
print(geo_stats)

# ============================================================================
# 9. ADVANCED VISUALIZATIONS
# ============================================================================

def create_visualizations(df_master):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Delivery Rate by Country
    country_data = df_master.groupby('Country').agg({
        'Nbr_Commande_Livree': 'sum',
        'Nbr_Commande_Non_Livree': 'sum'
    })
    country_data['Total'] = country_data.sum(axis=1)
    country_data['Delivery_Rate'] = (
        country_data['Nbr_Commande_Livree'] / country_data['Total'] * 100
    )
    country_data.nlargest(10, 'Total')['Delivery_Rate'].plot(
        kind='barh', ax=axes[0, 0], color='steelblue'
    )
    axes[0, 0].set_title('Top 10 Countries by Delivery Rate', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Delivery Rate (%)')
    
    # 2. Orders Over Time
    temporal = df_master.groupby('mois_annee')['id_seq_fait'].count()
    temporal.plot(kind='line', ax=axes[0, 1], marker='o', color='darkgreen')
    axes[0, 1].set_title('Order Trends Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Number of Orders')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Employee Performance Distribution
    emp_perf = df_master.groupby('Employee ID').agg({
        'Nbr_Commande_Livree': 'sum',
        'Nbr_Commande_Non_Livree': 'sum'
    })
    emp_perf['Total'] = emp_perf.sum(axis=1)
    emp_perf['Total'].plot(kind='hist', bins=20, ax=axes[1, 0], 
                           color='coral', edgecolor='black')
    axes[1, 0].set_title('Distribution of Orders per Employee', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Orders')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Delivered vs Undelivered
    delivered = df_master['Nbr_Commande_Livree'].sum()
    undelivered = df_master['Nbr_Commande_Non_Livree'].sum()
    axes[1, 1].pie([delivered, undelivered], 
                   labels=['Delivered', 'Undelivered'],
                   autopct='%1.1f%%', startangle=90,
                   colors=['#2ecc71', '#e74c3c'])
    axes[1, 1].set_title('Overall Delivery Status', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('overview_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

create_visualizations(df_master)

# ============================================================================
# 10. CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(df_master):
    """Analyze correlations between numerical variables"""
    
    # Select numerical columns
    numerical_cols = df_master.select_dtypes(include=[np.number]).columns
    corr_matrix = df_master[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, fmt='.2f')
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

corr_matrix = correlation_analysis(df_master)

# ============================================================================
# 11. RFM ANALYSIS (Recency, Frequency, Monetary)
# ============================================================================

def rfm_analysis(df_master):
    """Customer segmentation using RFM analysis"""
    
    # Detect available date columns and use the best one
    date_columns = ['Order_Date', 'date', 'Date', 'Annee', 'Year']
    date_col = None
    
    for col in date_columns:
        if col in df_master.columns:
            date_col = col
            break
    
    if date_col is None:
        print("Warning: No date column found. Using 'Annee' for recency calculation")
        date_col = 'Annee'
    
    # Calculate snapshot date (most recent date in dataset)
    try:
        snapshot_date = df_master[date_col].max()
    except:
        snapshot_date = df_master['Annee'].max()
        date_col = 'Annee'
    
    print(f"Using '{date_col}' for recency calculation")
    print(f"Snapshot date: {snapshot_date}")
    
    # Calculate RFM metrics
    rfm = df_master.groupby('Customer ID').agg({
        date_col: lambda x: snapshot_date - x.max(),  # Recency
        'id_seq_fait': 'count',  # Frequency
        'Nbr_Commande_Livree': 'sum'  # Monetary (using delivered orders as proxy)
    }).reset_index()
    
    rfm.columns = ['Customer_ID', 'Recency', 'Frequency', 'Monetary']
    
    print(f"\nRFM Summary Statistics:")
    print(rfm.describe())
    
    # Create RFM scores (1-5) with duplicate handling
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    except (ValueError, TypeError):
        # If qcut fails, use cut with manual bins or percentile-based scoring
        try:
            rfm['R_Score'] = pd.cut(rfm['Recency'], 
                                    bins=5, 
                                    labels=[5, 4, 3, 2, 1], 
                                    duplicates='drop')
        except:
            # Last resort: use rank-based scoring
            rfm['R_Score'] = 6 - pd.qcut(rfm['Recency'].rank(method='first'), 
                                         5, 
                                         labels=[1, 2, 3, 4, 5],
                                         duplicates='drop').astype(int)
    
    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, 
                                 labels=[1, 2, 3, 4, 5], 
                                 duplicates='drop')
    except (ValueError, TypeError):
        try:
            rfm['F_Score'] = pd.cut(rfm['Frequency'], 
                                    bins=5, 
                                    labels=[1, 2, 3, 4, 5], 
                                    duplicates='drop')
        except:
            # Use percentile-based scoring
            rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 
                                     q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=[1, 2, 3, 4, 5],
                                     duplicates='drop')
    
    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 5, 
                                 labels=[1, 2, 3, 4, 5], 
                                 duplicates='drop')
    except (ValueError, TypeError):
        try:
            rfm['M_Score'] = pd.cut(rfm['Monetary'], 
                                    bins=5, 
                                    labels=[1, 2, 3, 4, 5], 
                                    duplicates='drop')
        except:
            rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), 
                                     q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=[1, 2, 3, 4, 5],
                                     duplicates='drop')
    
    # Handle any NaN scores
    rfm['R_Score'] = rfm['R_Score'].fillna(3).astype(int)
    rfm['F_Score'] = rfm['F_Score'].fillna(3).astype(int)
    rfm['M_Score'] = rfm['M_Score'].fillna(3).astype(int)
    
    # Calculate RFM score
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    # Segment customers
    try:
        rfm['Segment'] = pd.cut(rfm['RFM_Score'], 
                                bins=[0, 6, 9, 12, 15],
                                labels=['At Risk', 'Developing', 'Loyal', 'Champions'],
                                include_lowest=True)
    except:
        # Alternative: Use quartiles for segmentation
        rfm['Segment'] = pd.qcut(rfm['RFM_Score'],
                                 q=4,
                                 labels=['At Risk', 'Developing', 'Loyal', 'Champions'],
                                 duplicates='drop')
    
    # Handle any remaining NaN in Segment
    rfm['Segment'] = rfm['Segment'].fillna('Developing')
    
    print(f"\nRFM Segmentation Complete!")
    print(f"Segments distribution:")
    print(rfm['Segment'].value_counts())
    
    return rfm

rfm_data = rfm_analysis(df_master)
print("\nRFM SEGMENTATION:")
print(rfm_data['Segment'].value_counts())

# ============================================================================
# 12. PREDICTIVE MODELING (Delivery Success)
# ============================================================================

def build_delivery_prediction_model(df_master):
    """Build a model to predict delivery success"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    # Prepare features
    df_model = df_master.copy()
    
    # Create target variable (1 if delivered, 0 if not)
    df_model['Delivered_Success'] = (
        df_model['Nbr_Commande_Livree'] > df_model['Nbr_Commande_Non_Livree']
    ).astype(int)
    
    # Select features
    feature_cols = ['Employee ID', 'Region', 'Territory', 'Country', 
                    'City', 'Annee', 'mois_annee']
    
    # Encode categorical variables
    le_dict = {}
    for col in feature_cols:
        if df_model[col].dtype == 'object':
            le = LabelEncoder()
            df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # Select encoded features
    encoded_features = [col + '_encoded' if df_model[col].dtype == 'object' 
                       else col for col in feature_cols]
    
    X = df_model[encoded_features].fillna(0)
    y = df_model['Delivered_Success']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluation
    print("\nMODEL PERFORMANCE:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': encoded_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Feature Importance for Delivery Success', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, feature_importance

model, feature_imp = build_delivery_prediction_model(df_master)

# ============================================================================
# 13. COHORT ANALYSIS
# ============================================================================

def cohort_analysis(df_master):
    """Analyze customer retention and behavior over time"""
    
    # Assuming you have proper date columns
    # df_master['Order_Date'] = pd.to_datetime(df_master['date_column'])
    
    # Create cohort month (first order month for each customer)
    df_master['Cohort'] = df_master.groupby('Customer ID')['Annee'].transform('min')
    
    # Count orders by cohort and period
    cohort_data = df_master.groupby(['Cohort', 'Annee']).agg({
        'Customer ID': 'nunique'
    }).reset_index()
    
    # Pivot for heatmap
    cohort_pivot = cohort_data.pivot(index='Cohort', 
                                      columns='Annee', 
                                      values='Customer ID')
    
    # Calculate retention rates
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(retention, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Customer Retention Rate by Cohort (%)', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Cohort (First Order Year)')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig('cohort_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return retention

retention_rates = cohort_analysis(df_master)

# ============================================================================
# 14. ANOMALY DETECTION
# ============================================================================

def detect_anomalies(df_master):
    """Detect unusual patterns in orders"""
    
    from scipy import stats

    # Calculate z-scores for delivered orders
    df_master['Delivered_ZScore'] = stats.zscore(
        df_master['Nbr_Commande_Livree'].fillna(0)
    )
    
    # Flag anomalies (z-score > 3 or < -3)
    df_master['Is_Anomaly'] = (
        abs(df_master['Delivered_ZScore']) > 3
    )
    
    anomalies = df_master[df_master['Is_Anomaly']]
    
    print(f"\nANOMALIES DETECTED: {len(anomalies)}")
    print("\nAnomaly Summary:")
    print(anomalies[['Customer ID', 'Employee ID', 'Country', 
                     'Nbr_Commande_Livree', 'Delivered_ZScore']].head(10))
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.scatter(df_master.index, df_master['Nbr_Commande_Livree'], 
                c=df_master['Is_Anomaly'], cmap='coolwarm', alpha=0.6)
    plt.title('Anomaly Detection in Delivered Orders', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Number of Delivered Orders')
    plt.xlabel('Order Index')
    plt.colorbar(label='Anomaly')
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return anomalies

anomalies = detect_anomalies(df_master)

# ============================================================================
# 15. EXPORT RESULTS
# ============================================================================

def export_analysis_results(customer_stats, employee_perf, geo_stats, rfm_data):
    """Export all analysis results to CSV"""
    
    customer_stats.to_csv('customer_analysis.csv', index=False)
    employee_perf.to_csv('employee_performance.csv', index=False)
    geo_stats.to_csv('geographical_analysis.csv', index=False)
    rfm_data.to_csv('rfm_segmentation.csv', index=False)
    
    print("\nAll analysis results exported successfully!")
    print("Files created:")
    print("- customer_analysis.csv")
    print("- employee_performance.csv")
    print("- geographical_analysis.csv")
    print("- rfm_segmentation.csv")

export_analysis_results(customer_stats, employee_perf, geo_stats, rfm_data)

# ============================================================================
# 16. COMPREHENSIVE REPORT GENERATOR
# ============================================================================

def generate_executive_summary(df_master):
    """Generate an executive summary report"""
    
    summary = f"""
    {'='*70}
    EXECUTIVE SUMMARY - ADVANCED DATA ANALYSIS
    {'='*70}
    
    OVERVIEW:
    - Total Orders: {len(df_master):,}
    - Total Delivered: {df_master['Nbr_Commande_Livree'].sum():,}
    - Total Undelivered: {df_master['Nbr_Commande_Non_Livree'].sum():,}
    - Overall Delivery Rate: {(df_master['Nbr_Commande_Livree'].sum() / (df_master['Nbr_Commande_Livree'].sum() + df_master['Nbr_Commande_Non_Livree'].sum()) * 100):.2f}%
    
    CUSTOMER INSIGHTS:
    - Unique Customers: {df_master['Customer ID'].nunique():,}
    - Countries Served: {df_master['Country'].nunique()}
    - Cities Served: {df_master['City'].nunique()}
    
    EMPLOYEE INSIGHTS:
    - Active Employees: {df_master['Employee ID'].nunique()}
    - Regions Covered: {df_master['Region'].nunique()}
    - Territories: {df_master['Territory'].nunique()}
    
    TOP PERFORMING COUNTRY:
    {df_master.groupby('Country')['id_seq_fait'].count().nlargest(1).to_string()}
    
    {'='*70}
    """
    
    print(summary)
    
    # Save to file
    with open('executive_summary.txt', 'w') as f:
        f.write(summary)

generate_executive_summary(df_master)

# ============================================================================
# QUICK START GUIDE
# ============================================================================

print("""
╔════════════════════════════════════════════════════════════════════╗
║          JUPYTER NOTEBOOK - ADVANCED ANALYSIS READY!               ║
╚════════════════════════════════════════════════════════════════════╝

STEP 1: Load your CSV files
   Uncomment lines 22-25 and adjust the file paths

STEP 2: Run data exploration
   Uncomment lines 52-55

STEP 3: Create master dataset
   Uncomment line 85

STEP 4: Run analyses (uncomment as needed):
   - KPIs: Line 108
   - Customer Analysis: Line 131
   - Employee Performance: Line 166
   - Temporal Analysis: Line 189
   - Geographical Analysis: Line 219
   - Visualizations: Line 311
   - RFM Analysis: Line 373
   - Predictive Model: Line 463
   - Anomaly Detection: Line 549

STEP 5: Export results
   Uncomment line 577

Happy Analyzing! 📊
""")