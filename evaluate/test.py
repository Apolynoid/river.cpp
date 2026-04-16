import sys
import time

from river import datasets
from river import metrics
from river import linear_model
from river import rules
from river import forest
from river import preprocessing

# 终极保险：拉高 Python 递归深度，防止极深树结构遍历时偶发的爆栈
sys.setrecursionlimit(5000)

def evaluate_pa_regressor():
    print("--- Running PARegressor (Baseline) ---")
    dataset = datasets.TrumpApproval()
    
    # PA 是纯线性模型，必须缩放
    scaler = preprocessing.StandardScaler()
    model = linear_model.PARegressor()
    mae = metrics.MAE()
    
    start_time = time.perf_counter()
    for step, (x, y) in enumerate(dataset, 1):
        scaler.learn_one(x)
        x_scaled = scaler.transform_one(x)
        
        y_pred = model.predict_one(x_scaled)
        if y_pred is not None:
            mae.update(y, y_pred)
            
        model.learn_one(x_scaled, y)
        
        if step % 500 == 0:
            print(f"[PARegressor] Step {step:4d} MAE: {mae.get():.4f}")
            
    return mae.get(), time.perf_counter() - start_time


def evaluate_am_rules():
    print("\n--- Running AMRules ---")
    dataset = datasets.TrumpApproval()
    
    # 修复：AMRules 的规则叶子节点默认带有 Adaptive 线性回归器
    # 必须加缩放镇压，否则遭遇大数值特征会原地爆炸！
    scaler = preprocessing.StandardScaler()
    model = rules.AMRules()
    mae = metrics.MAE()
    
    start_time = time.perf_counter()
    for step, (x, y) in enumerate(dataset, 1):
        scaler.learn_one(x)
        x_scaled = scaler.transform_one(x)
        
        y_pred = model.predict_one(x_scaled)
        if y_pred is not None:
            mae.update(y, y_pred)
            
        model.learn_one(x_scaled, y)
        
        if step % 500 == 0:
            print(f"[AMRules]     Step {step:4d} MAE: {mae.get():.4f}")
            
    return mae.get(), time.perf_counter() - start_time


def evaluate_arf_regressor():
    print("\n--- Running Adaptive Random Forest (ARF) ---")
    dataset = datasets.TrumpApproval()
    
    # 修复：ARF 同样使用 Adaptive 叶子节点，必须加缩放
    # 使用用户指定的正确 API 路径: forest.ARFRegressor
    scaler = preprocessing.StandardScaler()
    model = forest.ARFRegressor(seed=42)
    mae = metrics.MAE()
    
    start_time = time.perf_counter()
    for step, (x, y) in enumerate(dataset, 1):
        scaler.learn_one(x)
        x_scaled = scaler.transform_one(x)
        
        y_pred = model.predict_one(x_scaled)
        if y_pred is not None:
            mae.update(y, y_pred)
            
        model.learn_one(x_scaled, y)
        
        if step % 500 == 0:
            print(f"[ARFRegressor] Step {step:4d} MAE: {mae.get():.4f}")
            
    return mae.get(), time.perf_counter() - start_time


def main():
    print("Initializing Robust Benchmark for Trump Approval Rating...\n")
    print("Dataset Info: 1001 samples, tracking approval percentage.")
    print("-" * 60)
    
    # 彻底的扁平化调用，物理隔离每个模型的内存和调用栈
    pa_mae, pa_time = evaluate_pa_regressor()
    am_mae, am_time = evaluate_am_rules()
    arf_mae, arf_time = evaluate_arf_regressor()

    # 打印最终计分板
    print("\n" + "=" * 60)
    print(f"{'FINAL LEADERBOARD':^60}")
    print("=" * 60)
    print(f"{'Model':<20} | {'Final MAE (%)':<15} | {'Total Time (s)'}")
    print("-" * 60)
    print(f"{'PARegressor':<20} | {pa_mae:<15.4f} | {pa_time:.4f}")
    print(f"{'AMRules':<20} | {am_mae:<15.4f} | {am_time:.4f}")
    print(f"{'ARFRegressor':<20} | {arf_mae:<15.4f} | {arf_time:.4f}")
    print("=" * 60)
    
    print("\n* MAE represents the average absolute error in approval percentage points.")

if __name__ == "__main__":
    main()