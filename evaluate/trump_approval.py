import time
from river import rules, preprocessing, compose, metrics, drift, stream

DATA_PATH = '../data/trump_approval.csv'

def main():
    model = compose.Pipeline(preprocessing.StandardScaler(), rules.AMRules())

    metric = metrics.MSE()

    start_time = time.time()
    
    count = 0

    for x, y in stream.iter_csv(
        DATA_PATH, 
        target="five_thirty_eight", 
        converters={
            "ordinal_date": int,
            "gallup": float,
            "ipsos": float,
            "morning_consult": float,
            "rasmussen": float,
            "you_gov": float,
            "five_thirty_eight": float,
        },
    ):
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
        model.learn_one(x, y)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"Final MSE: {metric.get():.6f}")
    print(f"Time measured: {elapsed_ms:.0f} ms.")

if __name__ == "__main__":
    main()