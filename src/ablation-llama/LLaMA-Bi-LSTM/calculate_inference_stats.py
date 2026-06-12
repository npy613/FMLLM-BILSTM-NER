#!/usr/bin/env python3
import re
import numpy as np
from pathlib import Path


def parse_log_file(file_path):
    metrics = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        match = re.search(r'平均每个样本推理时间 \(排除warm-up\): ([\d.]+) ms', content)
        if match:
            metrics['avg_time'] = float(match.group(1))

        match = re.search(r'P50推理时延: ([\d.]+) ms', content)
        if match:
            metrics['p50'] = float(match.group(1))

        match = re.search(r'P95推理时延: ([\d.]+) ms', content)
        if match:
            metrics['p95'] = float(match.group(1))

        match = re.search(r'P99推理时延: ([\d.]+) ms', content)
        if match:
            metrics['p99'] = float(match.group(1))

    return metrics


def main():
    base_dir = Path(__file__).parent
    log_files = [base_dir / f"test_with_m&t_log{i}.txt" for i in range(1, 6)]

    all_metrics = {
        'avg_time': [],
        'p50': [],
        'p95': [],
        'p99': []
    }

    for log_file in log_files:
        if not log_file.exists():
            print(f"警告: 文件不存在 - {log_file}")
            continue

        metrics = parse_log_file(log_file)
        print(f"  {log_file.name}: 已解析")

        for key in all_metrics.keys():
            if key in metrics:
                all_metrics[key].append(metrics[key])

    print("\n" + "="*60)
    print("推理时间指标统计结果")
    print("="*60)

    for metric_name, values in all_metrics.items():
        if not values:
            print(f"\n{metric_name}: 无数据")
            continue

        values_array = np.array(values)
        mean_val = np.mean(values_array)
        std_val = np.std(values_array, ddof=1)  

        metric_display_names = {
            'avg_time': '平均每个样本推理时间 (排除warm-up)',
            'p50': 'P50推理时延',
            'p95': 'P95推理时延',
            'p99': 'P99推理时延'
        }

        print(f"\n{metric_display_names[metric_name]}:")
        print(f"  Mean: {mean_val:.2f} ms")
        print(f"  Std:  {std_val:.2f} ms")
        print(f"  数据点: {values}")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
