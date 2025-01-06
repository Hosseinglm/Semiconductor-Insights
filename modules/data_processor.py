import pandas as pd
import numpy as np
from typing import Tuple, Dict


class DataProcessor:
    def __init__(self):
        self.data = None

    def load_data(self, uploaded_files: Dict) -> pd.DataFrame:
        """Load and combine data from uploaded files"""
        dfs = []

        for file_type, file in uploaded_files.items():
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            else:
                continue

            df['source'] = file_type
            dfs.append(df)

        if not dfs:
            raise ValueError("No valid files uploaded")

        self.data = pd.concat(dfs, ignore_index=True)
        return self.data

    def validate_data(self) -> Tuple[bool, str]:
        """Validate data consistency across sources"""
        if self.data is None:
            return False, "No data loaded"

        sources = self.data['source'].unique()
        if len(sources) < 2:
            return False, "Need data from multiple sources for validation"

        # Check schema consistency
        columns_consistent = len(set(
            tuple(sorted(self.data[self.data['source'] == source].columns))
            for source in sources
        )) == 1

        if not columns_consistent:
            return False, "Inconsistent columns across sources"

        return True, "Data validation successful"

    def clean_data(self) -> pd.DataFrame:
        """Clean and handle missing data"""
        if self.data is None:
            raise ValueError("No data loaded")

        # Handle missing values
        self.data = self.data.fillna({
            'production_yield': self.data['production_yield'].mean(),
            'defect_rate': self.data['defect_rate'].mean(),
            'machine_downtime': self.data['machine_downtime'].mean(),
            'temperature': self.data['temperature'].mean(),
            'humidity': self.data['humidity'].mean()
        })

        # Remove duplicates
        self.data = self.data.drop_duplicates()

        # Convert date column
        self.data['date'] = pd.to_datetime(self.data['date'])

        return self.data

    def engineer_features(self) -> pd.DataFrame:
        """Create additional features"""
        if self.data is None:
            raise ValueError("No data loaded")

        # Calculate defects per unit
        self.data['defects_per_unit'] = (
                self.data['defect_rate'] * self.data['production_yield'] / 100
        )

        # Calculate downtime ratio (%)
        total_hours = 24
        self.data['downtime_ratio'] = (
                self.data['machine_downtime'] / total_hours * 100
        )

        # Add month and year
        self.data['month'] = self.data['date'].dt.month
        self.data['year'] = self.data['date'].dt.year

        return self.data

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics"""
        if self.data is None:
            raise ValueError("No data loaded")

        stats = {
            'avg_yield_by_machine': self.data.groupby('machine_id')['production_yield'].mean(),
            'avg_defect_by_material': self.data.groupby('material_type')['defect_rate'].mean(),
            'avg_downtime_by_machine': self.data.groupby('machine_id')['machine_downtime'].mean(),
            'environmental_corr': self.data[['temperature', 'humidity', 'production_yield']].corr()
        }

        return stats
