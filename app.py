import streamlit as st
import pandas as pd
from modules.data_processor import DataProcessor
from modules.ml_models import MLModels
from modules.visualizations import Visualizer
import json

# Page config
st.set_page_config(
    page_title="Semiconductor Manufacturing Analysis",
    page_icon="🏭",
    layout="wide"
)

# Initialize modules
data_processor = DataProcessor()
ml_models = MLModels()
visualizer = Visualizer()

def main():
    st.title("Semiconductor Manufacturing Analysis Dashboard")

    # Sidebar
    st.sidebar.header("Data Upload")
    production_file = st.sidebar.file_uploader(
        "Upload Production Data (CSV/JSON)",
        type=['csv', 'json']
    )
    performance_file = st.sidebar.file_uploader(
        "Upload Performance Data (CSV/JSON)",
        type=['csv', 'json']
    )
    quality_file = st.sidebar.file_uploader(
        "Upload Quality Data (CSV/JSON)",
        type=['csv', 'json']
    )

    if all([production_file, performance_file, quality_file]):
        try:
            data = data_processor.load_data({
                'production': production_file,
                'performance': performance_file,
                'quality': quality_file
            })

            valid, message = data_processor.validate_data()
            if not valid:
                st.error(message)
                return

            data = data_processor.clean_data()
            data = data_processor.engineer_features()
            stats = data_processor.get_summary_stats()

            # Display tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Overview", "Time Series Analysis",
                "Performance Analysis", "Machine Learning"
            ])

            # Overview Tab
            with tab1:
                display_overview(data, stats)

            # Time Series Analysis Tab
            with tab2:
                display_time_series(data)

            # Performance Analysis Tab
            with tab3:
                display_performance_analysis(data)

            # ML Tab
            with tab4:
                display_ml_section(data)

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload all required data files to begin analysis")

def display_overview(data: pd.DataFrame, stats: dict):
    st.header("Manufacturing Overview")

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Production Yield",
            f"{data['production_yield'].mean():.2f}%"
        )
    with col2:
        st.metric(
            "Average Defect Rate",
            f"{data['defect_rate'].mean():.2f}%"
        )
    with col3:
        st.metric(
            "Average Downtime",
            f"{data['machine_downtime'].mean():.2f} hours"
        )

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    st.plotly_chart(
        visualizer.create_correlation_heatmap(data),
        use_container_width=True
    )

    # Export data
    st.subheader("Export Analysis")
    if st.button("Download Processed Data"):
        processed_data = data.to_csv(index=False)
        st.download_button(
            "Download CSV",
            processed_data,
            "processed_data.csv",
            "text/csv"
        )

def display_time_series(data: pd.DataFrame):
    st.header("Time Series Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        metric = st.selectbox(
            "Select Metric to Analyze",
            ['production_yield', 'defect_rate', 'machine_downtime'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col2:
        st.markdown("### Analysis Type")
        analysis_type = st.radio(
            "",
            ["Overall Trend", "Individual Machines"],
            horizontal=True
        )

    if analysis_type == "Individual Machines":
        machines = data['machine_id'].unique().tolist()
        selected_machines = st.multiselect(
            "Select Machines to Display (max 5)",
            machines,
            default=machines[:3],
            max_selections=5
        )
        if selected_machines:
            data = data[data['machine_id'].isin(selected_machines)]

    # Display explanation
    with st.expander("📊 About This Chart"):
        st.markdown(f"""
        - The solid blue line shows the overall average {metric.replace('_', ' ')}
        - The shaded area represents one standard deviation from the mean
        - Individual machine trends can be toggled from the legend
        - Use the range slider below the chart to zoom into specific time periods
        """)

    st.plotly_chart(
        visualizer.create_time_series(data, metric),
        use_container_width=True
    )

def display_performance_analysis(data: pd.DataFrame):
    st.header("Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Performance")
        st.plotly_chart(
            visualizer.create_machine_comparison(data),
            use_container_width=True
        )

    with col2:
        st.subheader("Material Analysis")
        st.plotly_chart(
            visualizer.create_material_analysis(data),
            use_container_width=True
        )

def display_ml_section(data: pd.DataFrame):
    st.header("Machine Learning Models")

    ml_tab1, ml_tab2 = st.tabs(["Yield Prediction", "Defect Classification"])

    with ml_tab1:
        st.subheader("Yield Prediction Model")

        # Model explanation
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            This model predicts the production yield based on:
            - Temperature
            - Humidity
            - Machine Downtime

            The model uses Linear Regression with standardized features for better prediction accuracy.
            """)

        with st.form("yield_prediction_form"):
            temp = st.slider("Temperature (°C)", 20.0, 30.0, 25.0)
            humidity = st.slider("Humidity (%)", 30.0, 50.0, 40.0)
            downtime = st.slider("Downtime (hours)", 0.0, 24.0, 2.0)

            predict_button = st.form_submit_button("Train and Predict")

            if predict_button:
                try:
                    # Train model first
                    with st.spinner("Training model..."):
                        metrics = ml_models.train_yield_predictor(data)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Model Performance")
                            metrics_df = pd.DataFrame({
                                'Metric': ['R² (Train)', 'R² (Test)', 'MAE (Train)', 'MAE (Test)'],
                                'Value': [
                                    f"{metrics['r2_train']:.3f}",
                                    f"{metrics['r2_test']:.3f}",
                                    f"{metrics['mae_train']:.3f}",
                                    f"{metrics['mae_test']:.3f}"
                                ]
                            })
                            st.dataframe(metrics_df, hide_index=True)

                        with col2:
                            if 'feature_importance' in metrics:
                                st.write("### Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': ['Temperature', 'Humidity', 'Downtime'],
                                    'Importance': metrics['feature_importance']
                                })
                                st.bar_chart(importance_df.set_index('Feature'))

                    # Make prediction
                    with st.spinner("Making prediction..."):
                        prediction = ml_models.predict_yield(temp, humidity, downtime)
                        st.success(f"Predicted Yield: {prediction:.2f}%")

                        # Show prediction explanation
                        st.write("### Prediction Explanation")
                        explanation = ml_models.explain_yield_prediction(
                            temp, humidity, downtime
                        )
                        for factor, impact in explanation.items():
                            st.write(f"- {factor}: {impact}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with ml_tab2:
        st.subheader("Defect Risk Classification")

        # Model explanation
        with st.expander("ℹ️ About This Model"):
            st.markdown("""
            This model classifies defect risk as High or Low based on:
            - Temperature
            - Humidity
            - Production Yield
            - Machine Downtime

            The model uses Random Forest Classifier for better accuracy and feature importance analysis.
            """)

        with st.form("defect_prediction_form"):
            temp = st.slider(
                "Temperature (°C)",
                20.0, 30.0, 25.0,
                key="defect_temp"
            )
            humidity = st.slider(
                "Humidity (%)",
                30.0, 50.0, 40.0,
                key="defect_humidity"
            )
            yield_val = st.slider(
                "Production Yield (%)",
                70.0, 100.0, 85.0
            )
            downtime = st.slider(
                "Downtime (hours)",
                0.0, 24.0, 2.0,
                key="defect_downtime"
            )

            predict_button = st.form_submit_button("Train and Predict")

            if predict_button:
                try:
                    # Train model first
                    with st.spinner("Training model..."):
                        metrics = ml_models.train_defect_classifier(data)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Model Performance")
                            metrics_df = pd.DataFrame({
                                'Metric': [
                                    'Accuracy (Train)',
                                    'Accuracy (Test)',
                                    'Precision',
                                    'Recall',
                                    'F1 Score'
                                ],
                                'Value': [
                                    f"{metrics['accuracy_train']:.3f}",
                                    f"{metrics['accuracy_test']:.3f}",
                                    f"{metrics.get('precision', 0):.3f}",
                                    f"{metrics.get('recall', 0):.3f}",
                                    f"{metrics.get('f1', 0):.3f}"
                                ]
                            })
                            st.dataframe(metrics_df, hide_index=True)

                        with col2:
                            if 'feature_importance' in metrics:
                                st.write("### Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': [
                                        'Temperature', 'Humidity',
                                        'Production Yield', 'Downtime'
                                    ],
                                    'Importance': metrics['feature_importance']
                                })
                                st.bar_chart(importance_df.set_index('Feature'))

                    # Make prediction
                    with st.spinner("Making prediction..."):
                        risk_prob = ml_models.predict_defect_risk_proba(
                            temp, humidity, yield_val, downtime
                        )
                        risk = "High" if risk_prob > 0.5 else "Low"

                        # Create a color-coded probability indicator
                        prob_color = "red" if risk_prob > 0.7 else (
                            "orange" if risk_prob > 0.5 else "green"
                        )
                        st.markdown(f"""
                        ### Risk Assessment
                        - **Risk Level:** {risk}
                        - **Confidence:** <span style='color:{prob_color}'>{risk_prob*100:.1f}%</span>
                        """, unsafe_allow_html=True)

                        # Show prediction explanation
                        st.write("### Risk Factors")
                        explanation = ml_models.explain_defect_prediction(
                            temp, humidity, yield_val, downtime
                        )
                        for factor, impact in explanation.items():
                            st.write(f"- {factor}: {impact}")

                        # Display confusion matrix if available
                        if 'confusion_matrix' in metrics:
                            st.write("### Confusion Matrix")
                            cm = metrics['confusion_matrix']
                            cm_df = pd.DataFrame(
                                cm,
                                index=['Actual Low', 'Actual High'],
                                columns=['Predicted Low', 'Predicted High']
                            )
                            st.dataframe(cm_df)

                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()