"""
Gradio Interface for Commit Tracking System
Interactive web interface for commit tracking and optimization management
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging

# Import our modules
from commit_tracker import (
    CommitTracker, OptimizationCommit, CommitType, CommitStatus,
    create_commit_tracker
)
from version_manager import (
    VersionManager, VersionType, VersionStatus,
    create_version_manager
)
from optimization_registry import (
    OptimizationRegistry, OptimizationCategory, RegistryStatus,
    create_optimization_registry
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommitTrackingInterface:
    """Gradio interface for commit tracking system"""
    
    def __init__(self):
        # Initialize components
        self.commit_tracker = create_commit_tracker()
        self.version_manager = create_version_manager(use_wandb=False, use_tensorboard=False)
        self.optimization_registry = create_optimization_registry(auto_benchmark=False)
        
        # Create sample data
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        
        # Sample commits
        sample_commits = [
            OptimizationCommit(
                commit_id="demo_001",
                commit_hash="abc123",
                author="ML Engineer A",
                timestamp=datetime.now() - timedelta(days=5),
                message="Implement attention mechanism optimization",
                commit_type=CommitType.OPTIMIZATION,
                status=CommitStatus.COMPLETED,
                inference_time=45.2,
                memory_usage=2048,
                gpu_utilization=85.5,
                accuracy=0.923,
                loss=0.156,
                optimization_techniques=["attention_mechanism", "layer_norm"],
                hyperparameters={"learning_rate": 0.001, "batch_size": 32}
            ),
            OptimizationCommit(
                commit_id="demo_002",
                commit_hash="def456",
                author="ML Engineer B",
                timestamp=datetime.now() - timedelta(days=3),
                message="Add mixed precision training",
                commit_type=CommitType.TRAINING,
                status=CommitStatus.COMPLETED,
                inference_time=42.8,
                memory_usage=1536,
                gpu_utilization=92.3,
                accuracy=0.928,
                loss=0.142,
                optimization_techniques=["mixed_precision", "gradient_scaling"],
                hyperparameters={"learning_rate": 0.001, "batch_size": 64}
            ),
            OptimizationCommit(
                commit_id="demo_003",
                commit_hash="ghi789",
                author="ML Engineer C",
                timestamp=datetime.now() - timedelta(days=1),
                message="Implement model quantization",
                commit_type=CommitType.INFERENCE,
                status=CommitStatus.COMPLETED,
                inference_time=28.5,
                memory_usage=1024,
                gpu_utilization=78.2,
                accuracy=0.925,
                loss=0.148,
                optimization_techniques=["quantization", "int8_inference"],
                hyperparameters={"quantization_bits": 8}
            )
        ]
        
        # Add commits
        for commit in sample_commits:
            self.commit_tracker.add_commit(commit)
    
    def get_commit_statistics(self) -> str:
        """Get commit statistics as formatted string"""
        stats = self.commit_tracker.get_performance_statistics()
        
        if not stats:
            return "No commits found"
        
        return f"""
## 📊 Commit Statistics

**Total Commits:** {stats['total_commits']}
**Commits with Metrics:** {stats['commits_with_metrics']}

### Performance Metrics
- **Average Inference Time:** {stats['average_inference_time']:.2f}ms
- **Average Memory Usage:** {stats['average_memory_usage']:.2f}MB
- **Average GPU Utilization:** {stats['average_gpu_utilization']:.2f}%
- **Average Accuracy:** {stats['average_accuracy']:.3f}

### Best Performance
- **Best Accuracy:** {stats['best_accuracy']:.3f}
- **Fastest Inference:** {stats['fastest_inference']:.2f}ms
- **Lowest Memory:** {stats['lowest_memory']:.2f}MB
"""
    
    def get_commits_table(self) -> pd.DataFrame:
        """Get commits as DataFrame"""
        commits = self.commit_tracker.commits
        
        if not commits:
            return pd.DataFrame()
        
        data = []
        for commit in commits:
            data.append({
                'Commit ID': commit.commit_id,
                'Author': commit.author,
                'Message': commit.message[:50] + "..." if len(commit.message) > 50 else commit.message,
                'Type': commit.commit_type.value,
                'Status': commit.status.value,
                'Inference Time (ms)': commit.inference_time or 0,
                'Memory (MB)': commit.memory_usage or 0,
                'GPU Util (%)': commit.gpu_utilization or 0,
                'Accuracy': commit.accuracy or 0,
                'Date': commit.timestamp.strftime('%Y-%m-%d %H:%M')
            })
        
        return pd.DataFrame(data)
    
    def create_performance_chart(self) -> go.Figure:
        """Create performance visualization"""
        commits = self.commit_tracker.commits
        
        if not commits:
            return go.Figure()
        
        # Extract data
        commit_ids = [c.commit_id for c in commits]
        inference_times = [c.inference_time or 0 for c in commits]
        accuracies = [c.accuracy or 0 for c in commits]
        memory_usage = [c.memory_usage or 0 for c in commits]
        
        # Create subplot
        fig = go.Figure()
        
        # Add inference time trace
        fig.add_trace(go.Scatter(
            x=commit_ids,
            y=inference_times,
            mode='lines+markers',
            name='Inference Time (ms)',
            yaxis='y'
        ))
        
        # Add accuracy trace
        fig.add_trace(go.Scatter(
            x=commit_ids,
            y=[a * 100 for a in accuracies],  # Convert to percentage
            mode='lines+markers',
            name='Accuracy (%)',
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Performance Metrics Over Time',
            xaxis_title='Commit ID',
            yaxis=dict(title='Inference Time (ms)', side='left'),
            yaxis2=dict(title='Accuracy (%)', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        return fig
    
    def create_memory_chart(self) -> go.Figure:
        """Create memory usage visualization"""
        commits = self.commit_tracker.commits
        
        if not commits:
            return go.Figure()
        
        # Extract data
        commit_ids = [c.commit_id for c in commits]
        memory_usage = [c.memory_usage or 0 for c in commits]
        gpu_utilization = [c.gpu_utilization or 0 for c in commits]
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=commit_ids,
            y=memory_usage,
            name='Memory Usage (MB)',
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            x=commit_ids,
            y=gpu_utilization,
            name='GPU Utilization (%)',
            marker_color='lightcoral',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Memory Usage and GPU Utilization',
            xaxis_title='Commit ID',
            yaxis=dict(title='Memory Usage (MB)', side='left'),
            yaxis2=dict(title='GPU Utilization (%)', side='right', overlaying='y'),
            barmode='group'
        )
        
        return fig
    
    def add_commit(self, 
                   commit_id: str,
                   author: str,
                   message: str,
                   commit_type: str,
                   inference_time: float,
                   memory_usage: float,
                   gpu_utilization: float,
                   accuracy: float,
                   loss: float) -> str:
        """Add a new commit"""
        
        try:
            # Create commit
            commit = OptimizationCommit(
                commit_id=commit_id,
                commit_hash=f"hash_{commit_id}",
                author=author,
                timestamp=datetime.now(),
                message=message,
                commit_type=CommitType(commit_type),
                status=CommitStatus.COMPLETED,
                inference_time=inference_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                accuracy=accuracy,
                loss=loss,
                optimization_techniques=["custom_optimization"]
            )
            
            # Add to tracker
            self.commit_tracker.add_commit(commit)
            
            return f"✅ Successfully added commit {commit_id}"
            
        except Exception as e:
            return f"❌ Error adding commit: {str(e)}"
    
    def get_optimization_recommendations(self, commit_id: str) -> str:
        """Get optimization recommendations for a commit"""
        
        commit = self.commit_tracker.get_commit(commit_id)
        if not commit:
            return f"❌ Commit {commit_id} not found"
        
        recommendations = self.commit_tracker.get_optimization_recommendations(commit)
        
        if not recommendations:
            return "No specific recommendations available"
        
        result = f"## 💡 Optimization Recommendations for {commit_id}\n\n"
        for i, rec in enumerate(recommendations, 1):
            result += f"{i}. {rec}\n"
        
        return result
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        with gr.Blocks(title="Commit Tracking System", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# 🚀 TruthGPT Commit Tracking System")
            gr.Markdown("Advanced commit tracking with performance analytics and optimization recommendations")
            
            with gr.Tabs():
                
                # Dashboard Tab
                with gr.Tab("📊 Dashboard"):
                    
                    with gr.Row():
                        with gr.Column():
                            stats_output = gr.Markdown()
                            refresh_stats = gr.Button("🔄 Refresh Statistics", variant="secondary")
                        
                        with gr.Column():
                            commits_table = gr.DataFrame(
                                headers=["Commit ID", "Author", "Message", "Type", "Status", 
                                       "Inference Time (ms)", "Memory (MB)", "GPU Util (%)", 
                                       "Accuracy", "Date"],
                                interactive=False
                            )
                            refresh_table = gr.Button("🔄 Refresh Table", variant="secondary")
                    
                    # Performance Charts
                    with gr.Row():
                        performance_chart = gr.Plot(label="Performance Metrics")
                        memory_chart = gr.Plot(label="Memory Usage")
                    
                    # Event handlers
                    refresh_stats.click(
                        fn=self.get_commit_statistics,
                        outputs=stats_output
                    )
                    
                    refresh_table.click(
                        fn=self.get_commits_table,
                        outputs=commits_table
                    )
                    
                    # Load initial data
                    interface.load(
                        fn=self.get_commit_statistics,
                        outputs=stats_output
                    )
                    
                    interface.load(
                        fn=self.get_commits_table,
                        outputs=commits_table
                    )
                    
                    interface.load(
                        fn=self.create_performance_chart,
                        outputs=performance_chart
                    )
                    
                    interface.load(
                        fn=self.create_memory_chart,
                        outputs=memory_chart
                    )
                
                # Add Commit Tab
                with gr.Tab("➕ Add Commit"):
                    
                    with gr.Row():
                        with gr.Column():
                            commit_id_input = gr.Textbox(
                                label="Commit ID",
                                placeholder="e.g., opt_001",
                                value="demo_004"
                            )
                            author_input = gr.Textbox(
                                label="Author",
                                placeholder="e.g., ML Engineer",
                                value="Demo User"
                            )
                            message_input = gr.Textbox(
                                label="Message",
                                placeholder="e.g., Implement new optimization",
                                value="Demo optimization commit"
                            )
                            commit_type_input = gr.Dropdown(
                                choices=[t.value for t in CommitType],
                                label="Commit Type",
                                value=CommitType.OPTIMIZATION.value
                            )
                        
                        with gr.Column():
                            inference_time_input = gr.Number(
                                label="Inference Time (ms)",
                                value=35.0
                            )
                            memory_usage_input = gr.Number(
                                label="Memory Usage (MB)",
                                value=1200
                            )
                            gpu_utilization_input = gr.Number(
                                label="GPU Utilization (%)",
                                value=85.0
                            )
                            accuracy_input = gr.Number(
                                label="Accuracy",
                                value=0.92,
                                minimum=0.0,
                                maximum=1.0
                            )
                            loss_input = gr.Number(
                                label="Loss",
                                value=0.15
                            )
                    
                    add_button = gr.Button("➕ Add Commit", variant="primary")
                    add_output = gr.Markdown()
                    
                    add_button.click(
                        fn=self.add_commit,
                        inputs=[
                            commit_id_input, author_input, message_input,
                            commit_type_input, inference_time_input,
                            memory_usage_input, gpu_utilization_input,
                            accuracy_input, loss_input
                        ],
                        outputs=add_output
                    )
                
                # Recommendations Tab
                with gr.Tab("💡 Recommendations"):
                    
                    with gr.Row():
                        with gr.Column():
                            commit_id_select = gr.Dropdown(
                                choices=[],
                                label="Select Commit ID",
                                interactive=True
                            )
                            get_recommendations_button = gr.Button("🔍 Get Recommendations", variant="primary")
                        
                        with gr.Column():
                            recommendations_output = gr.Markdown()
                    
                    # Update commit ID choices
                    def update_commit_choices():
                        commits = self.commit_tracker.commits
                        return [c.commit_id for c in commits]
                    
                    get_recommendations_button.click(
                        fn=self.get_optimization_recommendations,
                        inputs=commit_id_select,
                        outputs=recommendations_output
                    )
                    
                    # Load commit choices
                    interface.load(
                        fn=update_commit_choices,
                        outputs=commit_id_select
                    )
                
                # Analytics Tab
                with gr.Tab("📈 Analytics"):
                    
                    gr.Markdown("## Advanced Analytics")
                    
                    with gr.Row():
                        analytics_output = gr.Markdown()
                        refresh_analytics = gr.Button("🔄 Refresh Analytics", variant="secondary")
                    
                    def get_analytics():
                        commits = self.commit_tracker.commits
                        
                        if not commits:
                            return "No data available for analytics"
                        
                        # Calculate trends
                        inference_times = [c.inference_time for c in commits if c.inference_time]
                        accuracies = [c.accuracy for c in commits if c.accuracy]
                        
                        if len(inference_times) > 1:
                            inference_trend = "Improving" if inference_times[-1] < inference_times[0] else "Degrading"
                        else:
                            inference_trend = "Insufficient data"
                        
                        if len(accuracies) > 1:
                            accuracy_trend = "Improving" if accuracies[-1] > accuracies[0] else "Degrading"
                        else:
                            accuracy_trend = "Insufficient data"
                        
                        return f"""
## 📈 Analytics Report

### Performance Trends
- **Inference Time Trend:** {inference_trend}
- **Accuracy Trend:** {accuracy_trend}

### Statistical Summary
- **Total Commits:** {len(commits)}
- **Average Inference Time:** {np.mean(inference_times):.2f}ms
- **Average Accuracy:** {np.mean(accuracies):.3f}
- **Performance Variance:** {np.var(inference_times):.2f}

### Optimization Impact
- **Best Performing Commit:** {max(commits, key=lambda x: x.accuracy or 0).commit_id}
- **Fastest Inference:** {min(commits, key=lambda x: x.inference_time or float('inf')).commit_id}
- **Most Memory Efficient:** {min(commits, key=lambda x: x.memory_usage or float('inf')).commit_id}
"""
                    
                    refresh_analytics.click(
                        fn=get_analytics,
                        outputs=analytics_output
                    )
                    
                    interface.load(
                        fn=get_analytics,
                        outputs=analytics_output
                    )
        
        return interface

def launch_interface():
    """Launch the Gradio interface"""
    
    # Create interface
    interface = CommitTrackingInterface()
    app = interface.create_interface()
    
    # Launch
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    launch_interface()



