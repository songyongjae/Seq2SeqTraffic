import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from model import lstm_encoder_decoder
from transform import windowDataset
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, name, ratio, query, df, df_output, weighted, device, model, learning_rate, epochs, optimizer, criterion, scaler):
        self.name = name
        self.ratio = ratio
        self.query = query
        self.df = df
        self.df_output = df_output
        self.weighted = weighted
        self.device = device
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = scaler
        self.path = "pth"

    def train(self):
        for keyword in self.query:
            try:
                index = self.df.columns.get_loc(keyword)
                c_vector = self.df.iloc[:, index].to_numpy().reshape(-1, 1)
                c_vector = self.scaler.fit_transform(c_vector)

                iw = 144  # 3 days of hourly data
                ow = 72   # 3 days of half-hourly data
                train_dataset = windowDataset(c_vector, input_window=iw, output_window=ow, stride=1)
                train_loader = DataLoader(train_dataset, batch_size=64)

                self.model.train()
                with tqdm(range(self.epochs), desc=f"Training {self.name}: {keyword}") as tr:
                    for _ in tr:
                        total_loss = 0.0
                        for x, y in train_loader:
                            self.optimizer.zero_grad()
                            x, y = x.to(self.device, dtype=torch.float32), y.to(self.device, dtype=torch.float32)
                            output = self.model(x, y, ow, 0.6)
                            loss = self.criterion(output, y)
                            loss.backward()
                            self.optimizer.step()
                            total_loss += loss.item()
                        tr.set_postfix(loss=f"{total_loss / len(train_loader):.5f}")

                predict = self.model.predict(torch.tensor(c_vector[-iw:], dtype=torch.float32).to(self.device), target_len=ow)
                predict = self.scaler.inverse_transform(predict.reshape(-1, 1)).flatten()
                real = self.scaler.inverse_transform(train_dataset[0][1].reshape(-1, 1)).flatten()

                self._visualize_results(predict, real, keyword)
                self._save_results(predict, keyword)

            except KeyError:
                print(f"'{keyword}' column does not exist in the DataFrame.")
        
        self.df_output.to_excel(f"pred_{self.name}.xlsx", index=False)
        return self.weighted

    def _visualize_results(self, predict, real, keyword):
        plt.figure(figsize=(10, 6))
        plt.plot(predict, label='Prediction', color='orange')
        plt.plot(real, label='Actual', color='blue')
        plt.title(f"Predictions vs Actual: {keyword}", fontsize=16)
        plt.legend()
        plt.savefig(f"images-{self.name}/{keyword}_results.png")
        plt.show()

    def _save_results(self, predict, keyword):
        temp_df = pd.DataFrame({keyword: predict[-24:]})
        self.df_output = pd.concat([self.df_output, temp_df], axis=1)
        self.weighted = pd.concat([self.weighted, temp_df * self.ratio], axis=1)
        os.makedirs(self.path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.path, f"{self.name}_{keyword}.pth"))
