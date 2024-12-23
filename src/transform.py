import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Transform:
    def __init__(self, df1, df2, query, merged_df, basket, examples):
        self.df1 = df1
        self.df2 = df2
        self.query = query
        self.merged = merged_df
        self.basket = basket
        self.examples = examples

    def sum_vector(self):
        sum_vector = np.zeros(len(self.df1.columns) - 2)
        for col_idx in range(2, len(self.df1.columns)):
            sum_vector[col_idx - 2] = self.df1.iloc[-24:, col_idx].sum()
        for col_idx in range(2, len(self.df2.columns)):
            sum_vector[col_idx - 2] += self.df2.iloc[-24:, col_idx].sum()
        sum_vector = np.log(sum_vector) / np.log(np.mean(sum_vector))
        return sum_vector

    def adj(self):
        adj = np.zeros(len(self.query))
        sum_vector = self.sum_vector()

        for idx, keyword in enumerate(self.query):
            try:
                index = self.df1.columns.get_loc(keyword)
                adj[idx] = sum_vector[index - 2]
            except KeyError:
                print(f"'{keyword}' column not found in df1.")

        return adj

    def result_matrix(self):
        datas = self.merged.iloc[:, 1:].to_numpy()
        adj = self.adj()
        matrixs = np.zeros((24 // 4, len(self.query)))

        for j in range(len(self.query)):
            for i in range(0, len(datas) - 4, 4):
                chunk = datas[i:i + 4, j]
                maxs = np.max(chunk) / np.mean(chunk)
                mins = np.min(chunk) / np.mean(chunk)
                matrixs[i // 4, j] = maxs * adj[j] if abs(maxs - 1) > abs(mins - 1) else mins * adj[j]

        result_matrix = pd.DataFrame()
        result_matrix['날짜'] = pd.to_datetime(self.basket['날짜']).dt.strftime('%m/%d/%Y')

        for idx, keyword in enumerate(self.query):
            try:
                index = self.basket.columns.get_loc(keyword)
                result_matrix[keyword] = self.basket.iloc[:, index] * matrixs[:, idx]
            except KeyError:
                print(f"'{keyword}' column not found in basket.")

        return result_matrix
