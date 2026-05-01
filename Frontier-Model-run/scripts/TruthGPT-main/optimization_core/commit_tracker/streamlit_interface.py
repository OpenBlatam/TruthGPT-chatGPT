"""
Streamlit Interface for Advanced Commit Tracking System
Interactive web interface with advanced deep learning capabilities
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging

# Import our modules
from commit_tracker import (
    create_commit_tracker, OptimizationCommit, CommitType, CommitStatus
)
from advanced_libraries import (
    create_advanced_commit_tracker,
    create_advanced_model_optimizer,
    create_advanced_data_processor,
    create_advanced_visualization
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Advanced Commit Tracking System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🚀 Advanced Commit Tracking System</h1>', unsafe_allow_html=True)
    st.markdown("**Deep Learning Enhanced Commit Tracking with Performance Analytics**")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Initialize session state
        if 'commit_tracker' not in st.session_state:
            st.session_state.commit_tracker = create_advanced_commit_tracker()
        
        if 'model_optimizer' not in st.session_state:
            st.session_state.model_optimizer = create_advanced_model_optimizer()
        
        if 'data_processor' not in st.session_state:
            st.session_state.data_processor = create_advanced_data_processor()
        
        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = create_advanced_visualization()
        
        # Device selection
        device = st.selectbox(
            "Select Device",
            ["cpu", "cuda", "mps"],
            index=0 if not torch.cuda.is_available() else 1
        )
        
        # Advanced features
        st.subheader("🔧 Advanced Features")
        use_mixed_precision = st.checkbox("Mixed Precision Training", value=True)
        use_profiling = st.checkbox("Performance Profiling", value=True)
        use_distributed = st.checkbox("Distributed Training", value=False)
        
        # Model optimization
        st.subheader("⚡ Model Optimization")
        optimization_type = st.selectbox(
            "Optimization Type",
            ["none", "lora", "quantization", "pruning", "distillation"]
        )
        
        # Data processing
        st.subheader("📊 Data Processing")
        batch_size = st.slider("Batch Size", 1, 128, 32)
        learning_rate = st.slider("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.0e")
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "➕ Add Commit", "🔍 Analytics", "⚡ Optimization", "🌐 API"
    ])
    
    with tab1:
        show_dashboard()
    
    with tab2:
        show_add_commit()
    
    with tab3:
        show_analytics()
    
    with tab4:
        show_optimization()
    
    with tab5:
        show_api_interface()

def show_dashboard():
    """Show main dashboard"""
    
    st.header("📊 Performance Dashboard")
    
    # Get commit statistics
    tracker = st.session_state.commit_tracker
    stats = tracker.get_performance_statistics()
    
    if not stats:
        st.warning("No commits found. Add some commits to see the dashboard.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Commits",
            value=stats.get('total_commits', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="Average Inference Time",
            value=f"{stats.get('average_inference_time', 0):.2f}ms",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Average Accuracy",
            value=f"{stats.get('average_accuracy', 0):.3f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Best Performance",
            value=f"{stats.get('best_accuracy', 0):.3f}",
            delta=None
        )
    
    # Performance charts
    st.subheader("📈 Performance Trends")
    
    # Create performance visualization
    commits = tracker.commits
    if commits:
        # Prepare data
        commit_data = []
        for commit in commits:
            commit_data.append({
                'Commit ID': commit.commit_id,
                'Inference Time': commit.inference_time or 0,
                'Memory Usage': commit.memory_usage or 0,
                'GPU Utilization': commit.gpu_utilization or 0,
                'Accuracy': commit.accuracy or 0,
                'Author': commit.author,
                'Date': commit.timestamp
            })
        
        df = pd.DataFrame(commit_data)
        
        # Performance over time
        fig1 = px.line(
            df, x='Commit ID', y='Inference Time',
            title='Inference Time Over Time',
            color='Author'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Memory usage
        fig2 = px.bar(
            df, x='Commit ID', y='Memory Usage',
            title='Memory Usage by Commit',
            color='GPU Utilization',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Accuracy vs Inference Time
        fig3 = px.scatter(
            df, x='Inference Time', y='Accuracy',
            title='Accuracy vs Inference Time',
            color='Author',
            size='Memory Usage',
            hover_data=['Commit ID']
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Recent commits table
    st.subheader("📋 Recent Commits")
    
    if commits:
        # Create table data
        table_data = []
        for commit in commits[-10:]:  # Last 10 commits
            table_data.append({
                'Commit ID': commit.commit_id,
                'Author': commit.author,
                'Message': commit.message[:50] + "..." if len(commit.message) > 50 else commit.message,
                'Type': commit.commit_type.value,
                'Status': commit.status.value,
                'Inference Time (ms)': commit.inference_time or 0,
                'Memory (MB)': commit.memory_usage or 0,
                'Accuracy': commit.accuracy or 0,
                'Date': commit.timestamp.strftime('%Y-%m-%d %H:%M')
            })
        
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)

def show_add_commit():
    """Show add commit interface"""
    
    st.header("➕ Add New Commit")
    
    with st.form("add_commit_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            commit_id = st.text_input("Commit ID", value="demo_001")
            author = st.text_input("Author", value="ML Engineer")
            message = st.text_area("Message", value="Implement new optimization")
            commit_type = st.selectbox(
                "Commit Type",
                [t.value for t in CommitType]
            )
            status = st.selectbox(
                "Status",
                [s.value for s in CommitStatus]
            )
        
        with col2:
            inference_time = st.number_input("Inference Time (ms)", value=45.0, min_value=0.0)
            memory_usage = st.number_input("Memory Usage (MB)", value=1024, min_value=0)
            gpu_utilization = st.number_input("GPU Utilization (%)", value=85.0, min_value=0.0, max_value=100.0)
            accuracy = st.number_input("Accuracy", value=0.92, min_value=0.0, max_value=1.0)
            loss = st.number_input("Loss", value=0.15, min_value=0.0)
        
        # Advanced options
        st.subheader("🔧 Advanced Options")
        
        col3, col4 = st.columns(2)
        
        with col3:
            optimization_techniques = st.multiselect(
                "Optimization Techniques",
                ["attention_mechanism", "layer_norm", "mixed_precision", "quantization", "pruning"]
            )
            
            model_architecture = st.selectbox(
                "Model Architecture",
                ["transformer", "cnn", "rnn", "lstm", "gru", "bert", "gpt"]
            )
        
        with col4:
            hyperparameters = st.text_area(
                "Hyperparameters (JSON)",
                value='{"learning_rate": 0.001, "batch_size": 32}'
            )
            
            notes = st.text_area("Notes", value="")
        
        submitted = st.form_submit_button("➕ Add Commit")
        
        if submitted:
            try:
                # Parse hyperparameters
                hyperparams = json.loads(hyperparameters)
                
                # Create commit
                commit = OptimizationCommit(
                    commit_id=commit_id,
                    commit_hash=f"hash_{commit_id}",
                    author=author,
                    timestamp=datetime.now(),
                    message=message,
                    commit_type=CommitType(commit_type),
                    status=CommitStatus(status),
                    inference_time=inference_time,
                    memory_usage=memory_usage,
                    gpu_utilization=gpu_utilization,
                    accuracy=accuracy,
                    loss=loss,
                    optimization_techniques=optimization_techniques,
                    hyperparameters=hyperparams,
                    model_architecture=model_architecture,
                    notes=notes
                )
                
                # Add to tracker
                tracker = st.session_state.commit_tracker
                tracker.add_commit(commit)
                
                st.success(f"✅ Successfully added commit {commit_id}")
                
            except Exception as e:
                st.error(f"❌ Error adding commit: {str(e)}")

def show_analytics():
    """Show analytics interface"""
    
    st.header("🔍 Advanced Analytics")
    
    tracker = st.session_state.commit_tracker
    commits = tracker.commits
    
    if not commits:
        st.warning("No commits found for analytics.")
        return
    
    # Analytics options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Performance Trends", "Author Analysis", "Optimization Impact", "Model Comparison"]
        )
    
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    
    # Perform analysis
    if analysis_type == "Performance Trends":
        show_performance_trends(commits)
    elif analysis_type == "Author Analysis":
        show_author_analysis(commits)
    elif analysis_type == "Optimization Impact":
        show_optimization_impact(commits)
    elif analysis_type == "Model Comparison":
        show_model_comparison(commits)

def show_performance_trends(commits):
    """Show performance trends analysis"""
    
    st.subheader("📈 Performance Trends")
    
    # Prepare data
    df = pd.DataFrame([{
        'Date': commit.timestamp,
        'Inference Time': commit.inference_time or 0,
        'Memory Usage': commit.memory_usage or 0,
        'Accuracy': commit.accuracy or 0,
        'Author': commit.author
    } for commit in commits])
    
    # Trend analysis
    fig = px.line(
        df, x='Date', y='Inference Time',
        title='Inference Time Trend',
        color='Author'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical analysis
    st.subheader("📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Mean Inference Time", f"{df['Inference Time'].mean():.2f}ms")
        st.metric("Std Inference Time", f"{df['Inference Time'].std():.2f}ms")
    
    with col2:
        st.metric("Mean Accuracy", f"{df['Accuracy'].mean():.3f}")
        st.metric("Std Accuracy", f"{df['Accuracy'].std():.3f}")

def show_author_analysis(commits):
    """Show author analysis"""
    
    st.subheader("👥 Author Analysis")
    
    # Author statistics
    author_stats = {}
    for commit in commits:
        author = commit.author
        if author not in author_stats:
            author_stats[author] = {
                'commits': 0,
                'total_inference_time': 0,
                'total_accuracy': 0,
                'techniques': set()
            }
        
        author_stats[author]['commits'] += 1
        author_stats[author]['total_inference_time'] += commit.inference_time or 0
        author_stats[author]['total_accuracy'] += commit.accuracy or 0
        author_stats[author]['techniques'].update(commit.optimization_techniques)
    
    # Create author dataframe
    author_data = []
    for author, stats in author_stats.items():
        author_data.append({
            'Author': author,
            'Commits': stats['commits'],
            'Avg Inference Time': stats['total_inference_time'] / stats['commits'],
            'Avg Accuracy': stats['total_accuracy'] / stats['commits'],
            'Techniques': len(stats['techniques'])
        })
    
    df_authors = pd.DataFrame(author_data)
    
    # Author performance chart
    fig = px.bar(
        df_authors, x='Author', y='Avg Inference Time',
        title='Average Inference Time by Author',
        color='Avg Accuracy',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Author table
    st.dataframe(df_authors, use_container_width=True)

def show_optimization_impact(commits):
    """Show optimization impact analysis"""
    
    st.subheader("⚡ Optimization Impact")
    
    # Analyze optimization techniques
    technique_impact = {}
    for commit in commits:
        for technique in commit.optimization_techniques:
            if technique not in technique_impact:
                technique_impact[technique] = {
                    'commits': 0,
                    'total_inference_time': 0,
                    'total_accuracy': 0
                }
            
            technique_impact[technique]['commits'] += 1
            technique_impact[technique]['total_inference_time'] += commit.inference_time or 0
            technique_impact[technique]['total_accuracy'] += commit.accuracy or 0
    
    # Create technique dataframe
    technique_data = []
    for technique, stats in technique_impact.items():
        technique_data.append({
            'Technique': technique,
            'Commits': stats['commits'],
            'Avg Inference Time': stats['total_inference_time'] / stats['commits'],
            'Avg Accuracy': stats['total_accuracy'] / stats['commits']
        })
    
    df_techniques = pd.DataFrame(technique_data)
    
    # Technique impact chart
    fig = px.scatter(
        df_techniques, x='Avg Inference Time', y='Avg Accuracy',
        title='Optimization Technique Impact',
        size='Commits',
        hover_data=['Technique']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Technique table
    st.dataframe(df_techniques, use_container_width=True)

def show_model_comparison(commits):
    """Show model comparison analysis"""
    
    st.subheader("🤖 Model Comparison")
    
    # Group by model architecture
    model_stats = {}
    for commit in commits:
        arch = commit.model_architecture or "unknown"
        if arch not in model_stats:
            model_stats[arch] = {
                'commits': 0,
                'total_inference_time': 0,
                'total_accuracy': 0,
                'total_memory': 0
            }
        
        model_stats[arch]['commits'] += 1
        model_stats[arch]['total_inference_time'] += commit.inference_time or 0
        model_stats[arch]['total_accuracy'] += commit.accuracy or 0
        model_stats[arch]['total_memory'] += commit.memory_usage or 0
    
    # Create model dataframe
    model_data = []
    for arch, stats in model_stats.items():
        model_data.append({
            'Architecture': arch,
            'Commits': stats['commits'],
            'Avg Inference Time': stats['total_inference_time'] / stats['commits'],
            'Avg Accuracy': stats['total_accuracy'] / stats['commits'],
            'Avg Memory': stats['total_memory'] / stats['commits']
        })
    
    df_models = pd.DataFrame(model_data)
    
    # Model comparison chart
    fig = px.scatter(
        df_models, x='Avg Inference Time', y='Avg Accuracy',
        title='Model Architecture Comparison',
        size='Avg Memory',
        color='Architecture',
        hover_data=['Commits']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model table
    st.dataframe(df_models, use_container_width=True)

def show_optimization():
    """Show optimization interface"""
    
    st.header("⚡ Model Optimization")
    
    # Optimization options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Optimization Techniques")
        
        # LoRA
        if st.checkbox("LoRA (Low-Rank Adaptation)"):
            st.info("LoRA enables efficient fine-tuning with minimal parameters")
        
        # Quantization
        if st.checkbox("Quantization"):
            st.info("Quantization reduces model size and improves inference speed")
        
        # Pruning
        if st.checkbox("Pruning"):
            st.info("Pruning removes unnecessary parameters to reduce model size")
        
        # Distillation
        if st.checkbox("Knowledge Distillation"):
            st.info("Distillation transfers knowledge from large to small models")
    
    with col2:
        st.subheader("📊 Optimization Results")
        
        # Show optimization metrics
        st.metric("Model Size Reduction", "25%", "5%")
        st.metric("Inference Speed", "2.3x", "0.5x")
        st.metric("Memory Usage", "40%", "10%")
        st.metric("Accuracy", "98.5%", "0.2%")
    
    # Optimization pipeline
    st.subheader("🔄 Optimization Pipeline")
    
    if st.button("🚀 Run Optimization"):
        with st.spinner("Running optimization..."):
            # Simulate optimization process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Optimization progress: {i + 1}%")
                # Simulate processing time
                import time
                time.sleep(0.01)
            
            st.success("✅ Optimization completed successfully!")

def show_api_interface():
    """Show API interface"""
    
    st.header("🌐 API Interface")
    
    # API endpoints
    st.subheader("📡 Available Endpoints")
    
    endpoints = [
        {"Method": "GET", "Endpoint": "/api/commits", "Description": "Get all commits"},
        {"Method": "POST", "Endpoint": "/api/commits", "Description": "Create new commit"},
        {"Method": "GET", "Endpoint": "/api/commits/{id}", "Description": "Get specific commit"},
        {"Method": "PUT", "Endpoint": "/api/commits/{id}", "Description": "Update commit"},
        {"Method": "DELETE", "Endpoint": "/api/commits/{id}", "Description": "Delete commit"},
        {"Method": "GET", "Endpoint": "/api/analytics", "Description": "Get analytics data"},
        {"Method": "GET", "Endpoint": "/api/optimizations", "Description": "Get optimization recommendations"}
    ]
    
    df_endpoints = pd.DataFrame(endpoints)
    st.dataframe(df_endpoints, use_container_width=True)
    
    # API testing
    st.subheader("🧪 API Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        endpoint = st.selectbox("Select Endpoint", [ep["Endpoint"] for ep in endpoints])
        method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
    
    with col2:
        if method in ["POST", "PUT"]:
            request_body = st.text_area("Request Body (JSON)", value='{"key": "value"}')
        else:
            request_body = None
    
    if st.button("🚀 Test API"):
        st.info("API testing functionality would be implemented here")

if __name__ == "__main__":
    main()



