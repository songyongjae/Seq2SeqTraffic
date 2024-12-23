import os
import pandas as pd
from config import DEVICE, MODEL, CRITERION, SCALER, LEARNING_RATE, EPOCHS, OPTIMIZER, QUERY
from train import ModelTrainer
from transform import Transform

if __name__ == "__main__":
    # Load data
    df1 = pd.read_excel("google.xlsx")
    df2 = pd.read_excel("naver.xlsx")
    basket = pd.read_excel("basket.xlsx")

    # Initialize trainers for Google and Naver datasets
    trainer1 = ModelTrainer(
        name='Google',
        ratio=0.35,
        query=QUERY,
        df=df1,
        df_output=pd.DataFrame(),
        weighted=pd.DataFrame(),
        device=DEVICE,
        model=MODEL,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        scaler=SCALER
    )

    trainer2 = ModelTrainer(
        name='Naver',
        ratio=0.65,
        query=QUERY,
        df=df2,
        df_output=pd.DataFrame(),
        weighted=pd.DataFrame(),
        device=DEVICE,
        model=MODEL,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        scaler=SCALER
    )

    # Train models
    weighted1 = trainer1.train()
    weighted2 = trainer2.train()

    # Merge weighted results
    merged_df = weighted1.add(weighted2, fill_value=0)
    merged_df.to_excel("merged.xlsx", index=False)

    # Apply transformations and compute results
    transformer = Transform(df1, df2, QUERY, merged_df, basket, examples=[])
    result_matrix = transformer.result_matrix()
    result_matrix.to_excel("result.xlsx", index=False)
