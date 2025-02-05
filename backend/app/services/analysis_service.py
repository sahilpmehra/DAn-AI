from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import openai
from app.core.config import settings
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from app.services.file_service import FileService
import networkx as nx
from app.services.code_execution_service import CodeExecutionService
import json

class AnalysisService:
    def __init__(self, file_service: FileService = None):
        self.file_service = file_service or FileService()
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.operation_registry = {
            'statistical': self._statistical_analysis,
            'temporal': self._temporal_analysis,
            'comparative': self._comparative_analysis,
            'clustering': self._clustering_analysis
        }
        self.code_executor = CodeExecutionService()

    async def execute_analysis(self, analysis_plan: Dict, session_id: str) -> Dict[str, Any]:
        """Execute analysis based on the plan"""
        try:
            # Get the dataset
            df = await self.file_service.get_dataframe(session_id)
            metadata = await self.file_service.get_metadata(session_id)
            
            # Validate required columns
            # self._validate_columns(df, analysis_plan['required_columns'])
            
            # Build dependency graph
            graph = self._build_dependency_graph(analysis_plan['steps'])
            
            # Execute steps in order
            results = await self._execute_steps(graph, df)
            
            return {
                'results': results,
                'execution_trace': self._get_execution_trace(graph)
            }
            
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str]):
        """Validate that required columns exist in the dataset"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
    # Clean numeric columns
    def _clean_numeric_value(self,value):
        """Convert various string formats to numeric values"""
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float)):
            return value
        if not isinstance(value, str):
            return np.nan
        
        # Remove any commas and whitespace
        value = value.replace(',', '').strip()
        
        try:
            # Handle percentages
            if value.endswith('%'):
                return float(value.rstrip('%')) / 100
                
            # Handle currency (assuming $ but can be expanded)
            if value.startswith('$'):
                return float(value.lstrip('$'))
                
            # Handle K/M/B notation (e.g., "1.5M" -> 1500000)
            multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
            if value[-1] in multipliers:
                number = float(value[:-1])
                return number * multipliers[value[-1]]
                
            # Try direct conversion for any other numeric strings
            return float(value)
            
        except (ValueError, TypeError):
            return np.nan

    async def _statistical_analysis(self, df: pd.DataFrame, columns: List[str]) -> Dict:
        """Comprehensive statistical analysis"""
        results = {
            'basic_stats': {},
            'distributions': {},
            'correlations': {},
            'outliers': {}
        }

        # Clean numeric columns
        numeric_df = df[columns].apply(lambda x: pd.to_numeric(
            x.map(self._clean_numeric_value), 
            errors='coerce'
        ))

        # Convert NumPy types to Python native types
        results['basic_stats'] = {
            'summary': {k: {ki: float(vi) if pd.notnull(vi) else None 
                           for ki, vi in v.items()}
                       for k, v in numeric_df.describe().to_dict().items()},
            'skewness': {k: float(v) if pd.notnull(v) else None 
                         for k, v in numeric_df.skew().to_dict().items()},
            'kurtosis': {k: float(v) if pd.notnull(v) else None 
                         for k, v in numeric_df.kurtosis().to_dict().items()}
        }
        
        # Distribution analysis
        for col in columns:
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                # Shapiro-Wilk test for normality
                stat, p_value = stats.shapiro(numeric_df[col].dropna())
                results['distributions'][col] = {
                    'normality_test': {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': bool(p_value > 0.05)
                    }
                }
        
        # Correlation analysis
        if len(columns) > 1:
            # Convert correlation matrices to native Python types
            results['correlations'] = {
                'pearson': {
                    k: {k2: float(v2) if pd.notnull(v2) else None 
                        for k2, v2 in v.items()}
                    for k, v in numeric_df[columns].corr(method='pearson').to_dict().items()
                },
                'spearman': {
                    k: {k2: float(v2) if pd.notnull(v2) else None 
                        for k2, v2 in v.items()}
                    for k, v in numeric_df[columns].corr(method='spearman').to_dict().items()
                }
            }
        
        # Outlier detection
        for col in columns:
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                Q1 = float(numeric_df[col].quantile(0.25))
                Q3 = float(numeric_df[col].quantile(0.75))
                IQR = Q3 - Q1
                outliers = numeric_df[col][(numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)]
                results['outliers'][col] = {
                    'count': int(len(outliers)),
                    'values': [float(x) if pd.notnull(x) else None for x in outliers.tolist()]
                }
        
        return results

    async def _temporal_analysis(self, df: pd.DataFrame, time_column: str, value_columns: List[str]) -> Dict:
        """Time series analysis"""
        results = {
            'trends': {},
            'seasonality': {},
            'rolling_stats': {}
        }
        
        # Convert time column to datetime if needed
        df[time_column] = pd.to_datetime(df[time_column])
        
        for col in value_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Trend analysis
                results['trends'][col] = {
                    'rolling_mean': df[col].rolling(window=7).mean().tolist(),
                    'rolling_std': df[col].rolling(window=7).std().tolist()
                }
                
                # Seasonality detection
                if len(df) >= 2:
                    try:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        decomposition = seasonal_decompose(df[col], period=min(len(df), 7))
                        results['seasonality'][col] = {
                            'trend': decomposition.trend.tolist(),
                            'seasonal': decomposition.seasonal.tolist(),
                            'residual': decomposition.resid.tolist()
                        }
                    except Exception:
                        results['seasonality'][col] = None
                
                # Rolling statistics
                results['rolling_stats'][col] = {
                    'weekly_avg': df.resample('W', on=time_column)[col].mean().tolist(),
                    'monthly_avg': df.resample('M', on=time_column)[col].mean().tolist()
                }
        
        return results

    async def _comparative_analysis(self, df: pd.DataFrame, group_column: str, value_columns: List[str]) -> Dict:
        """Comparative analysis between groups"""
        results = {
            'group_stats': {},
            'statistical_tests': {}
        }
        
        # Group statistics
        results['group_stats'] = df.groupby(group_column)[value_columns].agg([
            'mean', 'median', 'std', 'count'
        ]).to_dict()
        
        # Statistical tests between groups
        for col in value_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                groups = [group for _, group in df.groupby(group_column)[col]]
                
                # ANOVA test if more than 2 groups
                if len(groups) > 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    results['statistical_tests'][col] = {
                        'test': 'ANOVA',
                        'statistic': f_stat,
                        'p_value': p_value
                    }
                # T-test if exactly 2 groups
                elif len(groups) == 2:
                    t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
                    results['statistical_tests'][col] = {
                        'test': 't-test',
                        'statistic': t_stat,
                        'p_value': p_value
                    }
        
        return results

    async def _clustering_analysis(self, df: pd.DataFrame, columns: List[str], n_clusters: int = 3) -> Dict:
        """Perform clustering analysis"""
        # Prepare data
        X = df[columns].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        return {
            'clusters': clusters.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'pca_components': X_pca.tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist()
        }

    async def _create_visualizations(self, df: pd.DataFrame, analysis_type: str, columns: List[str], **kwargs) -> List[Dict]:
        """Generate visualizations based on analysis type"""
        visualizations = []
        
        if analysis_type == 'distribution':
            # Distribution plots
            for col in columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Histogram with KDE
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=df[col], name='Histogram', nbinsx=30))
                    fig.add_trace(go.Violin(x=df[col], name='KDE'))
                    fig.update_layout(title=f'Distribution of {col}')
                    visualizations.append({
                        'type': 'distribution',
                        'data': fig.to_json()
                    })
        
        elif analysis_type == 'correlation':
            if len(columns) > 1:
                # Correlation heatmap
                corr_matrix = df[columns].corr()
                fig = px.imshow(corr_matrix,
                              labels=dict(color="Correlation"),
                              x=columns,
                              y=columns)
                visualizations.append({
                    'type': 'correlation_heatmap',
                    'data': fig.to_json()
                })
                
                # Scatter matrix
                fig = px.scatter_matrix(df[columns])
                visualizations.append({
                    'type': 'scatter_matrix',
                    'data': fig.to_json()
                })
        
        elif analysis_type == 'temporal':
            time_col = kwargs.get('time_column')
            if time_col:
                # Time series plot
                fig = go.Figure()
                for col in columns:
                    if col != time_col:
                        fig.add_trace(go.Scatter(x=df[time_col], y=df[col], name=col))
                fig.update_layout(title='Time Series Analysis')
                visualizations.append({
                    'type': 'time_series',
                    'data': fig.to_json()
                })
        
        elif analysis_type == 'clustering':
            if 'clusters' in kwargs and 'pca_components' in kwargs:
                # Cluster visualization
                pca_df = pd.DataFrame(kwargs['pca_components'], columns=['PC1', 'PC2'])
                pca_df['Cluster'] = kwargs['clusters']
                fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster')
                fig.update_layout(title='Cluster Analysis (PCA)')
                visualizations.append({
                    'type': 'clustering',
                    'data': fig.to_json()
                })
        
        return visualizations

    async def _generate_insights(self, results: Dict) -> List[str]:
        """Generate natural language insights from analysis results"""
        insights = []
        
        # Statistical insights
        if 'basic_stats' in results:
            for col, stats in results['basic_stats']['summary'].items():
                if 'mean' in stats and 'std' in stats:
                    insights.append(f"The average {col} is {stats['mean']:.2f} with a standard deviation of {stats['std']:.2f}")
        
        # Distribution insights
        if 'distributions' in results:
            for col, dist_info in results['distributions'].items():
                if 'normality_test' in dist_info:
                    is_normal = dist_info['normality_test']['is_normal']
                    insights.append(f"The distribution of {col} is {'normal' if is_normal else 'not normal'}")
        
        # Correlation insights
        if 'correlations' in results and 'pearson' in results['correlations']:
            correlations = results['correlations']['pearson']
            for col1 in correlations:
                for col2, corr in correlations[col1].items():
                    if col1 < col2 and abs(corr) > 0.5:
                        strength = 'strong' if abs(corr) > 0.7 else 'moderate'
                        direction = 'positive' if corr > 0 else 'negative'
                        insights.append(f"There is a {strength} {direction} correlation ({corr:.2f}) between {col1} and {col2}")
        
        return insights

    async def _generate_summary(self, results: Dict, plan: Dict) -> str:
        """Generate a natural language summary of the analysis"""
        # Use GPT-4 to generate a natural language summary
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a concise summary of the data analysis results. The summary should not be more than 100 words."
                },
                {
                    "role": "user",
                    "content": f"Results: {results}\nAnalysis Plan: {plan}"
                }
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content 

    def _build_dependency_graph(self, steps: List[Dict]) -> nx.DiGraph:
        """Build a directed graph of analysis steps"""
        graph = nx.DiGraph()
        
        for step in steps:
            graph.add_node(step['id'], step=step)
            for dep in step['depends_on']:
                graph.add_edge(dep, step['id'])
                
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Analysis plan contains circular dependencies")
            
        return graph

    async def _execute_steps(self, graph: nx.DiGraph, df: pd.DataFrame) -> Dict[int, Any]:
        """Execute analysis steps in topological order"""
        results = {}
        ordered_steps = list(nx.topological_sort(graph))
        
        for step_id in ordered_steps:
            step = graph.nodes[step_id]['step']
            try:
                # Get input data from dependencies
                input_data = self._get_dependency_data(step['depends_on'], results)
                
                # if step['operation_type'] == 'custom':
                #     # Generate and execute custom code for this step
                #     custom_code = await self._generate_custom_code(step, df.columns)
                #     result = self.code_executor.execute_code(custom_code, df)
                # else:
                #     # Execute predefined operation
                #     operation = self.operation_registry.get(step['operation'])
                #     if not operation:
                #         raise ValueError(f"Unknown predefined operation: {step['operation']}")
                #     result = await operation(df, **step['parameters'], input_data=input_data)

                # Generate and execute custom code for this step
                custom_code = await self._generate_custom_code(step, df.columns, input_data)
                result = self.code_executor.execute_code(custom_code, df)
                    
                results[step_id] = result
                
            except Exception as e:
                graph.nodes[step_id]['error'] = str(e)
                raise
                
        return results

    def _get_dependency_data(self, depends_on: List[int], results: Dict[int, Any]) -> Dict:
        """Get results from dependent steps"""
        return {dep: results[dep] for dep in depends_on if dep in results}

    def _get_execution_trace(self, graph: nx.DiGraph) -> List[Dict]:
        """Generate execution trace for debugging and visualization"""
        return [
            {
                'step_id': node,
                'description': graph.nodes[node]['step']['description'],
                'status': 'error' if 'error' in graph.nodes[node] else 'completed',
                'error': graph.nodes[node].get('error')
            }
            for node in graph.nodes
        ]

    async def _generate_custom_code(self, step: Dict, columns: List[str], input_data: Dict) -> str:
        """Generate custom code for a step"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """Generate Python code for data analysis.
                Use pandas, numpy, matplotlib, and seaborn only.
                The DataFrame is available as 'df'.
                Store results in:
                - result_data: for numerical/tabular results
                - plot_data: for visualization data
                - metadata: for additional information"""},
                {"role": "user", "content": f"""
                Generate code for the following analysis step:
                Description: {step['description']}
                Parameters: {json.dumps(step['parameters'])}
                Available columns: {', '.join(columns)}
                Previous results: {step['depends_on']}
                Input data: {input_data}"""}
            ]
        )
        
        return response.choices[0].message.content 