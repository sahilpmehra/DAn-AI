from typing import Dict, Any, List
import pandas as pd
import numpy as np
from app.services.file_service import FileService
from app.core.config import settings
import openai

class DataSummaryService:
    def __init__(self):
        self.file_service = FileService()
        # Initialize OpenAI client with your API key
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate_analysis_summary(self, df: pd.DataFrame, summary_stats: Dict) -> Dict[str, Any]:
        # Create a prompt for OpenAI that includes relevant dataset information
        stats_text = "\n".join([
            f"Column '{col}': Mean={stats.get('Mean', 'N/A'):.2f}, "
            f"Min={stats.get('Min', 'N/A'):.2f}, Max={stats.get('Max', 'N/A'):.2f}"
            for col, stats in summary_stats.items() if isinstance(stats.get('Mean'), (int, float))
        ])

        prompt = f"""Provide a concise summary (2-3 sentences) of the data and list 2-4 key variables that appear most important.
        Format the response as JSON with two fields:
        1. 'summary': A brief analysis of the data
        2. 'keyVariables': An array of the most important variable names
        
        Keep the summary focused on factual observations about the data structure and obvious patterns.
        
        Analyze this dataset with the following statistics:
        {stats_text}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a data analyst providing concise, factual summaries of datasets."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" },
                temperature=0.5
            )
            
            # Parse the JSON response
            ai_response = response.choices[0].message.content
            ai_analysis = eval(ai_response)  # Be careful with eval in production!
            
            # Calculate problematic variables based on null values
            total_rows = len(df)
            problematic_vars = [
                col for col in df.columns 
                if (df[col].isna().sum() / total_rows) > 0.2 or 
                (df[col].astype(str).str.strip().eq('').sum() / total_rows) > 0.2
            ]
            
            return {
                "summary": ai_analysis["summary"],
                "keyVariables": ai_analysis["keyVariables"],
                "problematicVariables": problematic_vars
            }
        except Exception as e:
            print(f"Error generating analysis summary: {str(e)}")
            return {
                "summary": "Unable to generate summary due to an error.",
                "keyVariables": [],
                "problematicVariables": []
            }

    async def get_summary_data(self, session_id: str) -> Dict[str, Any]:
        # # This is sample data matching the structure expected by the frontend
        # # In a real implementation, you would fetch this from your database
        # return {
        #     "preview_data": {
        #         "headers": ["Age", "Income", "Education", "Job_Satisfaction"],
        #         "data": [
        #             {"Age": 25, "Income": 45000, "Education": "Bachelor's", "Job_Satisfaction": 7},
        #             {"Age": 35, "Income": 65000, "Education": "Master's", "Job_Satisfaction": 8},
        #             {"Age": 28, "Income": 52000, "Education": "Bachelor's", "Job_Satisfaction": 6},
        #             {"Age": 42, "Income": 85000, "Education": "PhD", "Job_Satisfaction": 9},
        #             {"Age": 31, "Income": 58000, "Education": "Master's", "Job_Satisfaction": 7},
        #             {"Age": 25, "Income": 45000, "Education": "Bachelor's", "Job_Satisfaction": 7},
        #             {"Age": 35, "Income": 65000, "Education": "Master's", "Job_Satisfaction": 8},
        #             {"Age": 28, "Income": 52000, "Education": "Bachelor's", "Job_Satisfaction": 6},
        #             {"Age": 42, "Income": 85000, "Education": "PhD", "Job_Satisfaction": 9},
        #             {"Age": 31, "Income": 58000, "Education": "Master's", "Job_Satisfaction": 7},
        #             {"Age": 25, "Income": 45000, "Education": "Bachelor's", "Job_Satisfaction": 7},
        #             {"Age": 35, "Income": 65000, "Education": "Master's", "Job_Satisfaction": 8},
        #             {"Age": 28, "Income": 52000, "Education": "Bachelor's", "Job_Satisfaction": 6},
        #             {"Age": 42, "Income": 85000, "Education": "PhD", "Job_Satisfaction": 9},
        #             {"Age": 31, "Income": 58000, "Education": "Master's", "Job_Satisfaction": 7}
        #         ]
        #     },
        #     "summary_stats": {
        #         "headers": ["Metric", "Age", "Income", "Job_Satisfaction"],
        #         "data": [
        #             {"Metric": "Count", "Age": 5, "Income": 5, "Job_Satisfaction": 5},
        #             {"Metric": "Mean", "Age": 32.2, "Income": 61000, "Job_Satisfaction": 7.4},
        #             {"Metric": "Std", "Age": 6.5, "Income": 15166, "Job_Satisfaction": 1.14},
        #             {"Metric": "Min", "Age": 25, "Income": 45000, "Job_Satisfaction": 6},
        #             {"Metric": "Max", "Age": 42, "Income": 85000, "Job_Satisfaction": 9}
        #         ]
        #     },
        #     "analysis_summary": {
        #         "summary": "The dataset contains information about employees including their age, income, education level, and job satisfaction. The data shows a positive correlation between education level and income, with PhD holders earning the highest salaries.",
        #         "keyVariables": ["Income", "Education", "Job_Satisfaction"],
        #         "problematicVariables": ["Department", "Manager_ID"]
        #     }
        # }
        try:
            # Get the DataFrame from file_service
            df: pd.DataFrame = await self.file_service.get_dataframe(session_id)
            
            if df is None or df.empty:
                raise ValueError("No data available for analysis")

            # 1. Preview Data (first 10 rows)
            preview_df = df.head(10)
            preview_data = {
                "headers": list(df.columns),
                "data": preview_df.to_dict('records')
            }

            # 2. Summary Statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            summary_stats = {
                "headers": ["Metric"] + list(numeric_cols),
                "data": []
            }

            # Calculate statistics for numeric columns
            metrics = {
                "Count": df[numeric_cols].count(),
                "Mean": df[numeric_cols].mean().round(2),  # Round to 2 decimals
                "Std": df[numeric_cols].std().round(2),    # Round to 2 decimals
                "Min": df[numeric_cols].min().round(2),    # Round to 2 decimals
                "Max": df[numeric_cols].max().round(2),    # Round to 2 decimals
                "Missing": df[numeric_cols].isnull().sum()
            }

            # Format statistics into the required structure
            for metric_name, values in metrics.items():
                row = {"Metric": metric_name}
                for col in numeric_cols:
                    row[col] = values[col]
                summary_stats["data"].append(row)

            # Store column data types (for future use)
            dtypes = df.dtypes.to_dict()

            # 3. Analysis Summary
            analysis_summary = {'summary': 'The dataset contains multiple variables with a mix of categorical and numerical data. Observations suggest a correlation between certain key variables, indicating potential relationships that may warrant further investigation.', 'keyVariables': ['variable1', 'variable2', 'variable3', 'variable4'], 'problematicVariables': []}
            # analysis_summary = self.generate_analysis_summary(df, metrics)
            # problematicVariables = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.2]
            # analysis_summary["problematicVariables"] = problematicVariables            

            return {
                "preview_data": preview_data,
                "summary_stats": summary_stats,
                "analysis_summary": analysis_summary
            }

        except Exception as e:
            print(f"Error in get_summary_data: {str(e)}")
            raise ValueError(f"Failed to generate summary data: {str(e)}")
