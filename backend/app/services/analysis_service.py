from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from app.services.file_service import FileService

class AnalysisService:
    def __init__(self, file_service: FileService = None):
        self.file_service = file_service or FileService()

    async def execute_analysis(self, analysis_plan: Dict, session_id: str) -> Dict:
        """Execute the analysis plan and return results"""
        try:
            # Get the dataset using the session_id
            df = await self.file_service.get_dataframe(session_id)

            # Get metadata for additional context
            metadata = await self.file_service.get_metadata(session_id)
            
            # Validate required columns
            self._validate_columns(df, analysis_plan['required_columns'])

            # Execute analysis based on type
            results = {}
            for analysis_type in analysis_plan['analysis_type']:
                if analysis_type == 'statistical':
                    results.update(await self._statistical_analysis(df, analysis_plan['required_columns']))
                elif analysis_type == 'temporal':
                    results.update(await self._temporal_analysis(df, analysis_plan))
                elif analysis_type == 'comparative':
                    results.update(await self._comparative_analysis(df, analysis_plan))
                    
            print("Results: ", results)
            # Generate visualizations
            visualizations = await self._create_visualizations(df, analysis_plan, results)
            
            # Combine results
            return {
                'statistical_results': results,
                'visualizations': visualizations,
                'summary': await self._generate_summary(results, analysis_plan)
            }
            
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str]):
        """Validate that required columns exist in the dataset"""
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    async def _statistical_analysis(self, df: pd.DataFrame, columns: List[str]) -> Dict:
        """Comprehensive statistical analysis"""
        results = {
            'basic_stats': {},
            'distributions': {},
            'correlations': {},
            'outliers': {}
        }
        
        # Basic statistics
        results['basic_stats'] = {
            'summary': df[columns].describe().to_dict(),
            'skewness': df[columns].skew().to_dict(),
            'kurtosis': df[columns].kurtosis().to_dict()
        }
        
        # Distribution analysis
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Shapiro-Wilk test for normality
                stat, p_value = stats.shapiro(df[col].dropna())
                results['distributions'][col] = {
                    'normality_test': {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
                }
        
        # Correlation analysis
        if len(columns) > 1:
            results['correlations'] = {
                'pearson': df[columns].corr(method='pearson').to_dict(),
                'spearman': df[columns].corr(method='spearman').to_dict()
            }
        
        # Outlier detection using IQR method
        for col in columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[col][(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                results['outliers'][col] = {
                    'count': len(outliers),
                    'values': outliers.tolist()
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
                    "content": "Generate a concise summary of the data analysis results."
                },
                {
                    "role": "user",
                    "content": f"Results: {results}\nAnalysis Plan: {plan}"
                }
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content 