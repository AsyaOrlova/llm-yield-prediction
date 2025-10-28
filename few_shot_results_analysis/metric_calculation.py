import os

from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

class MetricCalculation:
    __high_yield_patterns = ['high-yielding', 'high yielding', 'high yield']


    @classmethod
    def calculate_aggregate_metrics(cls, df_path: str) -> dict:
        df = pd.read_csv(df_path)
        pred_columns = [col for col in df.columns if 'seed' in col]
        accuracy_list = []
        f1_list = []
        for pred_column in pred_columns:
            df = cls._preprocess_columns(df, pred_column)
            acc, f1 = cls._calculate_metrics_from_df(df, pred_column, 'high_yielding')
            accuracy_list.append(acc)
            f1_list.append(f1)

        return {'Accuracy': f'{np.round(np.mean(accuracy_list),2)}+-{np.round(np.std(accuracy_list),2)}',
                'F1-score': f'{np.round(np.mean(f1_list),2)}+-{np.round(np.std(f1_list),2)}'}


    @staticmethod
    def _calculate_metrics_from_df(df: pd.DataFrame, predicted_column:str, label_column: str) -> (float, float):
        acc = accuracy_score(df[label_column], df[predicted_column])
        f1 = f1_score(df[label_column], df[predicted_column])
        return acc, f1


    def _preprocess_columns(self, dataframe: pd.DataFrame, predicted_column: str) -> pd.DataFrame:

        answers = dataframe[predicted_column].astype(str).str.lower()
        high_yield_condition = answers.str.contains('|'.join(self.__high_yield_patterns), case=False, na=False)

        dataframe[predicted_column] = np.where(high_yield_condition, 0, 1)
        return dataframe


if __name__ == '__main__':
    result_folder = 'папка с резами'
    for file in os.listdir(result_folder):
        file_path = os.path.join(result_folder, file)
        results = MetricCalculation.calculate_aggregate_metrics(file_path)
        print(f'{file} - {results}')
