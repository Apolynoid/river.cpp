import time
from river import forest, preprocessing, compose, metrics, drift, stream

DATA_PATH = '../data/covtype.data'
NUM_FEATURES = 54
SEED = 42
N_MODELS = 10
MAX_FEATURES = 5

def main():
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        forest.ARFClassifier(
            n_models=N_MODELS,
            max_features=MAX_FEATURES,
            seed=SEED,
            drift_detector=drift.binary.DDM(),
            warning_detector=drift.binary.DDM(drift_threshold=2.0, warning_threshold=1.0),
        )
    )

    metric = metrics.Accuracy()

    column_names = [f"feat_{i}" for i in range(NUM_FEATURES)] + ["label"]
    
    converters = {name: float for name in column_names[:-1]}
    converters["label"] = int

    start_time = time.time()
    
    count = 0

    for x, y in stream.iter_csv(
        DATA_PATH, 
        target="label", 
        fieldnames=column_names,  # <--- 这里改了
        converters=converters
    ):
        y = y - 1  
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
        model.learn_one(x, y)

    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"Final Accuracy: {metric.get():.6f}")
    print(f"Time measured: {elapsed_ms:.0f} ms.")

if __name__ == "__main__":
    main()