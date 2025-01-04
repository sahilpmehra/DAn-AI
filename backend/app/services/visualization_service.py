from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class VisualizationService:
    def create_visualization(self, data: pd.DataFrame, viz_type: str, params: Dict[str, Any]) -> Dict:
        """Create visualization based on type and parameters"""
        if viz_type == "line":
            return self._create_line_plot(data, params)
        elif viz_type == "scatter":
            return self._create_scatter_plot(data, params)
        elif viz_type == "bar":
            return self._create_bar_plot(data, params)
        elif viz_type == "histogram":
            return self._create_histogram(data, params)
        elif viz_type == "box":
            return self._create_box_plot(data, params)
        elif viz_type == "heatmap":
            return self._create_heatmap(data, params)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")

    def _create_line_plot(self, data: pd.DataFrame, params: Dict) -> Dict:
        fig = px.line(
            data,
            x=params.get("x"),
            y=params.get("y"),
            title=params.get("title", "Line Plot"),
            color=params.get("color"),
            line_group=params.get("line_group")
        )
        return {"type": "line", "data": fig.to_json()}

    def _create_scatter_plot(self, data: pd.DataFrame, params: Dict) -> Dict:
        fig = px.scatter(
            data,
            x=params.get("x"),
            y=params.get("y"),
            title=params.get("title", "Scatter Plot"),
            color=params.get("color"),
            size=params.get("size"),
            trendline=params.get("trendline")
        )
        return {"type": "scatter", "data": fig.to_json()}

    def _create_bar_plot(self, data: pd.DataFrame, params: Dict) -> Dict:
        fig = px.bar(
            data,
            x=params.get("x"),
            y=params.get("y"),
            title=params.get("title", "Bar Plot"),
            color=params.get("color"),
            barmode=params.get("barmode", "group")
        )
        return {"type": "bar", "data": fig.to_json()}

    def _create_histogram(self, data: pd.DataFrame, params: Dict) -> Dict:
        fig = px.histogram(
            data,
            x=params.get("x"),
            nbins=params.get("nbins", 30),
            title=params.get("title", "Histogram"),
            color=params.get("color")
        )
        return {"type": "histogram", "data": fig.to_json()}

    def _create_box_plot(self, data: pd.DataFrame, params: Dict) -> Dict:
        fig = px.box(
            data,
            x=params.get("x"),
            y=params.get("y"),
            title=params.get("title", "Box Plot"),
            color=params.get("color")
        )
        return {"type": "box", "data": fig.to_json()}

    def _create_heatmap(self, data: pd.DataFrame, params: Dict) -> Dict:
        fig = px.imshow(
            data,
            title=params.get("title", "Heatmap"),
            color_continuous_scale=params.get("color_scale", "viridis")
        )
        return {"type": "heatmap", "data": fig.to_json()}

    def create_dashboard(self, data: pd.DataFrame, viz_configs: List[Dict]) -> Dict:
        """Create a dashboard with multiple visualizations"""
        rows = len(viz_configs)
        fig = make_subplots(rows=rows, cols=1, subplot_titles=[vc.get("title", "") for vc in viz_configs])
        
        for i, config in enumerate(viz_configs, 1):
            viz = self.create_visualization(data, config["type"], config["params"])
            viz_fig = go.Figure(viz["data"])
            for trace in viz_fig.data:
                fig.add_trace(trace, row=i, col=1)
        
        return {"type": "dashboard", "data": fig.to_json()}

    def export_visualization(self, viz_data: Dict, format: str = "json") -> Any:
        """Export visualization in different formats"""
        fig = go.Figure(viz_data["data"])
        
        if format == "json":
            return fig.to_json()
        elif format == "html":
            return fig.to_html()
        elif format == "image":
            return fig.to_image(format="png")
        else:
            raise ValueError(f"Unsupported export format: {format}") 