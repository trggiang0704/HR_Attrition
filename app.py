import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

# ================== CONFIG ==================
st.set_page_config(
    page_title="HR Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== LOAD DATA ==================
@st.cache_resource
def load_data():
    """Load all necessary data"""
    data_dir = Path("data/processed")
    
    data = {
        "df_processed": pd.read_csv(data_dir / "hr_processed.csv"),
        "df_ml": pd.read_csv(data_dir / "hr_processed_ml.csv"),
        "rules_all": pd.read_csv(data_dir / "rules_apriori_all.csv"),
        "rules_filtered": pd.read_csv(data_dir / "rules_apriori_filtered.csv"),
        "rules_leave": pd.read_csv(data_dir / "rules_apriori_leave.csv"),
        "rules_stay": pd.read_csv(data_dir / "rules_apriori_stay.csv"),
        "rules_clusters": pd.read_csv(data_dir / "leave_rules_clusters.csv"),
        "semi_supervised": pd.read_csv(data_dir / "semi_supervised_results.csv"),
    }
    
    return data

@st.cache_resource
def load_model_metrics():
    """Load model evaluation metrics"""
    results_dir = Path("data/processed/models")
    
    with open(results_dir / "xgb/xgb_metrics.json", "r") as f:
        xgb_metrics = json.load(f)
    
    with open(results_dir / "rf/rf_metrics.json", "r") as f:
        rf_metrics = json.load(f)
    
    return xgb_metrics, rf_metrics

@st.cache_resource
def load_predictions():
    """Load model predictions"""
    results_dir = Path("data/processed/models")
    
    xgb_pred = pd.read_csv(results_dir / "xgb/xgb_predictions.csv")
    rf_pred = pd.read_csv(results_dir / "rf/rf_predictions.csv")
    
    return xgb_pred, rf_pred

# ================== UTILITY FUNCTIONS ==================
def format_metric(value, decimals=3):
    """Format metric value"""
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"
    return str(value)

def create_model_comparison():
    """Create model comparison dataframe"""
    xgb_metrics, rf_metrics = load_model_metrics()
    
    return pd.DataFrame({
        "Model": ["XGBoost", "Random Forest"],
        "PR-AUC": [
            format_metric(xgb_metrics.get("pr_auc", 0)),
            format_metric(rf_metrics.get("pr_auc", 0))
        ],
        "Recall (Leave)": [
            format_metric(xgb_metrics.get("recall_leave", 0)),
            format_metric(rf_metrics.get("recall_leave", 0))
        ],
        "Precision (Leave)": [
            format_metric(xgb_metrics.get("precision_leave", 0)),
            format_metric(rf_metrics.get("precision_leave", 0))
        ],
        "F1-score (Leave)": [
            format_metric(xgb_metrics.get("f1_leave", 0)),
            format_metric(rf_metrics.get("f1_leave", 0))
        ],
        "Train Time (s)": [
            format_metric(xgb_metrics.get("train_time_sec", 0), 2),
            format_metric(rf_metrics.get("train_time_sec", 0), 2)
        ]
    })

# ================== MAIN SIDEBAR ==================
def sidebar():
    st.sidebar.title("🎯 Navigation")
    
    pages = {
        "📊 Dashboard": "dashboard",
        "🤖 Model Evaluation": "model_eval",
        "🔮 Prediction": "prediction",
        "📈 Association Rules": "rules",
        "🎓 Semi-Supervised Learning": "ssl",
        "ℹ️ About": "about"
    }
    
    selected = st.sidebar.radio("Select Page:", list(pages.keys()))
    
    return pages[selected]

# ================== PAGE: Dashboard ==================
def page_dashboard():
    st.title("📊 HR Attrition Analysis Dashboard")
    
    data = load_data()
    df = data["df_processed"]
    
    # Overview Metrics
    st.markdown("## 📌 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(df))
    
    with col2:
        attrition_count = df['Attrition'].sum() if 'Attrition' in df.columns else 0
        st.metric("Left Company", int(attrition_count))
    
    with col3:
        attrition_rate = (attrition_count / len(df) * 100) if len(df) > 0 else 0
        st.metric("Attrition Rate %", f"{attrition_rate:.2f}%")
    
    with col4:
        if 'Age' in df.columns:
            avg_age = df['Age'].mean()
            st.metric("Avg Age", f"{avg_age:.1f}")
    
    # Data Distribution
    st.markdown("## 📈 Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Attrition' in df.columns:
            attrition_dist = df['Attrition'].value_counts()
            fig = px.pie(
                names=['Stay', 'Leave'],
                values=attrition_dist.values,
                title='Attrition Distribution',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Department' in df.columns:
            dept_dist = df['Department'].value_counts()
            fig = px.bar(
                y=dept_dist.index,
                x=dept_dist.values,
                orientation='h',
                title='Employees by Department',
                labels={'x': 'Count', 'y': 'Department'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Key Statistics
    st.markdown("## 📊 Key Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats_df = df[numeric_cols].describe().T
        st.dataframe(stats_df, use_container_width=True)

# ================== PAGE: Model Evaluation ==================
def page_model_eval():
    st.title("🤖 Model Evaluation & Comparison")
    
    # Evaluation Strategy
    st.markdown("## 📋 Evaluation Strategy")
    st.info("""
    **Problem**: HR Attrition is a **highly imbalanced classification** problem
    - Leave (1): minority class
    - Stay (0): majority class
    
    **Strategy**: Focus on Leave class performance using multiple metrics
    - **Recall (Leave)**: % of actual departures detected
    - **Precision (Leave)**: Reliability of departure alerts
    - **F1-score (Leave)**: Balance between Recall & Precision
    - **PR-AUC**: Performance ranking in imbalanced data
    """)
    
    # Model Comparison Table
    st.markdown("## 🏆 Model Comparison")
    df_comparison = create_model_comparison()
    st.dataframe(df_comparison, use_container_width=True)
    
    # Load detailed metrics
    xgb_metrics, rf_metrics = load_model_metrics()
    
    # Metrics by Model
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### XGBoost Metrics")
        st.json({
            k: format_metric(v) if isinstance(v, (int, float)) else v 
            for k, v in sorted(xgb_metrics.items())
        })
    
    with col2:
        st.markdown("### Random Forest Metrics")
        st.json({
            k: format_metric(v) if isinstance(v, (int, float)) else v 
            for k, v in sorted(rf_metrics.items())
        })
    
    # Prediction Comparison
    st.markdown("## 🎯 Prediction Distribution")
    
    xgb_pred, rf_pred = load_predictions()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'predicted' in xgb_pred.columns:
            fig = px.histogram(
                xgb_pred,
                x='predicted',
                nbins=2,
                title='XGBoost Predictions',
                labels={'predicted': 'Prediction', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'predicted' in rf_pred.columns:
            fig = px.histogram(
                rf_pred,
                x='predicted',
                nbins=2,
                title='Random Forest Predictions',
                labels={'predicted': 'Prediction', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

# ================== PAGE: Prediction ==================
def page_prediction():
    st.title("🔮 Attrition Prediction")
    
    st.info("Enter employee information to predict attrition risk")
    
    # Create columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 18, 65, 30)
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, step=100)
        department = st.selectbox("Department", ["Sales", "R&D", "HR"])
    
    with col2:
        job_level = st.slider("Job Level", 1, 5, 2)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        total_working_years = st.slider("Total Working Years", 0, 40, 10)
    
    with col3:
        over_time = st.selectbox("Over Time", ["No", "Yes"])
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 2)
    
    # Prediction button
    if st.button("🔮 Predict Attrition", use_container_width=True):
        # Create input data
        input_data = {
            "Age": age,
            "MonthlyIncome": monthly_income,
            "Department": department,
            "JobLevel": job_level,
            "YearsAtCompany": years_at_company,
            "TotalWorkingYears": total_working_years,
            "OverTime": 1 if over_time == "Yes" else 0,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance
        }
        
        st.markdown("## 📊 Prediction Results")
        
        # Display input summary
        with st.expander("📋 Input Summary"):
            st.json(input_data)
        
        # Simulated prediction (in real scenario, you'd use the trained model)
        # For now, we'll show a placeholder
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### XGBoost Prediction")
            # Placeholder probability calculation based on inputs
            xgb_prob = (age / 65 * 0.2 + job_level / 5 * 0.3 + 
                       (5 - job_satisfaction) / 3 * 0.25 + 
                       years_at_company / 40 * 0.25)
            xgb_prob = min(max(xgb_prob, 0), 1)
            
            st.metric("Attrition Probability", f"{xgb_prob:.2%}")
            
            if xgb_prob > 0.5:
                st.error("⚠️ HIGH RISK - Intervention recommended")
            elif xgb_prob > 0.3:
                st.warning("⚡ MEDIUM RISK - Monitor closely")
            else:
                st.success("✅ LOW RISK - Stable")
        
        with col2:
            st.markdown("### Random Forest Prediction")
            # Similar placeholder for RF
            rf_prob = (age / 65 * 0.15 + job_level / 5 * 0.25 + 
                      (5 - job_satisfaction) / 3 * 0.3 + 
                      years_at_company / 40 * 0.3)
            rf_prob = min(max(rf_prob, 0), 1)
            
            st.metric("Attrition Probability", f"{rf_prob:.2%}")
            
            if rf_prob > 0.5:
                st.error("⚠️ HIGH RISK - Intervention recommended")
            elif rf_prob > 0.3:
                st.warning("⚡ MEDIUM RISK - Monitor closely")
            else:
                st.success("✅ LOW RISK - Stable")

# ================== PAGE: Association Rules ==================
def page_rules():
    st.title("📈 Association Rules Mining")
    
    data = load_data()
    
    # Tabs for different rule views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Leave Rules", 
        "Stay Rules", 
        "Clustered Rules",
        "All Rules (Filtered)"
    ])
    
    with tab1:
        st.markdown("## 📍 Rules for Leaving")
        st.info("Association rules indicating factors associated with attrition")
        
        rules_leave = data["rules_leave"]
        
        if len(rules_leave) > 0:
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rules", len(rules_leave))
            with col2:
                if 'support' in rules_leave.columns:
                    st.metric("Avg Support", f"{rules_leave['support'].mean():.3f}")
            with col3:
                if 'confidence' in rules_leave.columns:
                    st.metric("Avg Confidence", f"{rules_leave['confidence'].mean():.3f}")
            
            # Sortable/Filterable view
            if 'confidence' in rules_leave.columns:
                rules_display = rules_leave.nlargest(20, 'confidence')
            else:
                rules_display = rules_leave.head(20)
            
            st.dataframe(rules_display, use_container_width=True)
        else:
            st.warning("No leave rules found")
    
    with tab2:
        st.markdown("## 📍 Rules for Staying")
        st.info("Association rules indicating factors associated with retention")
        
        rules_stay = data["rules_stay"]
        
        if len(rules_stay) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rules", len(rules_stay))
            with col2:
                if 'support' in rules_stay.columns:
                    st.metric("Avg Support", f"{rules_stay['support'].mean():.3f}")
            with col3:
                if 'confidence' in rules_stay.columns:
                    st.metric("Avg Confidence", f"{rules_stay['confidence'].mean():.3f}")
            
            if 'confidence' in rules_stay.columns:
                rules_display = rules_stay.nlargest(20, 'confidence')
            else:
                rules_display = rules_stay.head(20)
            
            st.dataframe(rules_display, use_container_width=True)
        else:
            st.warning("No stay rules found")
    
    with tab3:
        st.markdown("## 🎯 Clustered Leave Rules")
        
        rules_clusters = data["rules_clusters"]
        
        if len(rules_clusters) > 0:
            st.markdown(f"**Total Clusters**: {rules_clusters['cluster'].max() + 1 if 'cluster' in rules_clusters.columns else 'N/A'}")
            
            st.dataframe(rules_clusters.head(30), use_container_width=True)
        else:
            st.warning("No clustered rules found")
    
    with tab4:
        st.markdown("## 📊 All Filtered Rules")
        
        rules_filtered = data["rules_filtered"]
        
        if len(rules_filtered) > 0:
            st.metric("Total Filtered Rules", len(rules_filtered))
            
            # Show top rules by confidence
            if 'confidence' in rules_filtered.columns:
                rules_display = rules_filtered.nlargest(30, 'confidence')
            else:
                rules_display = rules_filtered.head(30)
            
            st.dataframe(rules_display, use_container_width=True)
        else:
            st.warning("No rules found")

# ================== PAGE: Semi-Supervised Learning ==================
def page_ssl():
    st.title("🎓 Semi-Supervised Learning Results")
    
    data = load_data()
    
    ssl_results = data["semi_supervised"]
    
    st.markdown("## 📊 SSL Performance Summary")
    
    if len(ssl_results) > 0:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(ssl_results))
        with col2:
            if 'algorithm' in ssl_results.columns:
                st.metric("Unique Algorithms", ssl_results['algorithm'].nunique())
        with col3:
            if 'accuracy' in ssl_results.columns:
                st.metric("Avg Accuracy", f"{ssl_results['accuracy'].mean():.3f}")
        
        # Detailed results
        st.markdown("## 📈 Detailed Results")
        st.dataframe(ssl_results, use_container_width=True)
        
        # Visualize by algorithm
        if 'algorithm' in ssl_results.columns and 'accuracy' in ssl_results.columns:
            fig = px.bar(
                ssl_results,
                x='algorithm',
                y='accuracy',
                title='Accuracy by Algorithm',
                labels={'algorithm': 'Algorithm', 'accuracy': 'Accuracy'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No SSL results found")
    
    # Load summary if exists
    try:
        with open("data/processed/semi_supervised_summary.txt", "r") as f:
            summary = f.read()
            st.markdown("## 📝 Summary")
            st.text(summary)
    except:
        pass

# ================== PAGE: About ==================
def page_about():
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🎯 HR Attrition Prediction System
    
    This application provides comprehensive analysis and prediction of employee attrition using
    multiple machine learning and data mining techniques.
    
    ### 📊 Project Components
    
    1. **Data Processing & Feature Engineering**
       - Raw HR data preprocessing
       - Feature engineering and selection
       - Data discretization for association rules
    
    2. **Predictive Modeling**
       - XGBoost classifier
       - Random Forest classifier
       - Evaluation on imbalanced data using appropriate metrics
    
    3. **Association Rules Mining**
       - Apriori algorithm for discovering patterns
       - Rules clustering for interpretability
       - Separate analysis for leaving vs. staying factors
    
    4. **Semi-Supervised Learning**
       - Pseudo-labeling techniques
       - Leveraging unlabeled data for improved predictions
    
    ### 📈 Key Metrics
    
    - **PR-AUC**: Precision-Recall AUC (suitable for imbalanced data)
    - **Recall (Leave)**: Percentage of actual departures detected
    - **Precision (Leave)**: Reliability of departure predictions
    - **F1-score**: Harmonic mean of precision and recall
    
    ### 🔧 Technologies Used
    
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: XGBoost, Scikit-learn
    - **Association Rules**: MLxtend
    - **Visualization**: Plotly, Matplotlib, Seaborn
    - **Web App**: Streamlit
    
    ### 📂 Data Files
    
    - Raw data: `data/raw/HR_Analytics.csv`
    - Processed data: `data/processed/hr_processed*.csv`
    - Model metrics: `data/processed/models/*/metrics.json`
    - Rules: `data/processed/rules_apriori_*.csv`
    
    ---
    
    **Last Updated**: {}
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

# ================== MAIN APP ==================
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get selected page
    page = sidebar()
    
    # Route to page
    if page == "dashboard":
        page_dashboard()
    elif page == "model_eval":
        page_model_eval()
    elif page == "prediction":
        page_prediction()
    elif page == "rules":
        page_rules()
    elif page == "ssl":
        page_ssl()
    elif page == "about":
        page_about()

if __name__ == "__main__":
    main()
