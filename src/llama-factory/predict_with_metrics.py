#!/usr/bin/env python3
"""
LLaMA-Factory预测性能监控脚本
测量推理延迟和显存占用
支持流式预测以避免显存累积
"""
import os
import sys
import argparse

# 解析命令行参数（在导入torch之前）
def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA-Factory 预测性能监控")
    parser.add_argument('config', type=str, help='YAML 配置文件路径')
    parser.add_argument('device', type=int, nargs='?', default=0, help='GPU 设备 ID')
    parser.add_argument('--streaming', action='store_true',
                       help='使用流式预测以避免显存累积')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（仅用于流式预测）')
    return parser.parse_args()

# 必须在导入torch之前设置CUDA_VISIBLE_DEVICES
if __name__ == '__main__':
    args = parse_args()
    yaml_path = args.config
    device_id = args.device
    use_streaming = args.streaming
    batch_size = args.batch_size

    # 在导入torch之前设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

import time
import subprocess
import json
from pathlib import Path

def measure_prediction_performance(yaml_config_path):
    """
    运行LLaMA-Factory预测并测量性能指标

    Args:
        yaml_config_path: YAML配置文件路径
    """
    print("=" * 60)
    print("LLaMA-Factory 预测性能监控")
    print("=" * 60)

    # CUDA_VISIBLE_DEVICES已经在导入torch之前设置
    # 现在可以安全地使用cuda:0（映射到实际的GPU）

    # 获取初始显存占用（使用nvidia-smi查询实际GPU显存）
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', gpu_id],
            capture_output=True,
            text=True,
            check=True
        )
        initial_memory = float(result.stdout.strip())
        print(f"\n初始显存占用 (nvidia-smi): {initial_memory:.2f} MB")
    except Exception as e:
        print(f"\n警告: 无法获取初始显存占用: {e}")
        initial_memory = 0

    # 记录开始时间
    start_time = time.time()

    # 运行LLaMA-Factory预测
    print(f"\n开始预测...")
    print(f"配置文件: {yaml_config_path}")
    print(f"使用设备: GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")

    try:
        # 启动nvidia-smi监控（后台进程）
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        monitor_cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id} -l 1 > /tmp/gpu_monitor_{os.getpid()}.log &"
        subprocess.Popen(monitor_cmd, shell=True)

        # 调用llamafactory-cli
        cmd = ['llamafactory-cli', 'train', yaml_config_path]

        # 设置环境变量以跳过版本检查（如果datasets版本不匹配）
        env = os.environ.copy()
        env['DISABLE_VERSION_CHECK'] = '1'

        # 启动推理进程（非阻塞）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        # 等待第一个样本推理完成（通过监控显存变化来判断）
        print("等待第一个样本推理完成...")
        first_sample_memory = 0
        monitor_log = f"/tmp/gpu_monitor_{os.getpid()}.log"

        # 持续监控显存，直到显存稳定（模型加载完成）
        max_wait_time = 60  # 最多等待60秒
        check_interval = 2  # 每2秒检查一次
        stable_threshold = 50  # 显存变化小于50MB认为稳定

        last_memory = initial_memory
        stable_count = 0

        for i in range(int(max_wait_time / check_interval)):
            time.sleep(check_interval)

            try:
                result_mem = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', gpu_id],
                    capture_output=True,
                    text=True,
                    check=True
                )
                current_memory = float(result_mem.stdout.strip())
                memory_change = abs(current_memory - last_memory)

                print(f"  [{i*check_interval}s] 当前显存: {current_memory:.2f} MB (变化: {memory_change:.2f} MB)")

                # 如果显存变化小于阈值，认为稳定
                if memory_change < stable_threshold:
                    stable_count += 1
                    if stable_count >= 2:  # 连续2次稳定
                        first_sample_memory = current_memory
                        print(f"显存已稳定，第一个样本推理完成")
                        break
                else:
                    stable_count = 0

                last_memory = current_memory

            except Exception as e:
                print(f"警告: 无法获取显存: {e}")
                break

        if first_sample_memory == 0:
            # 如果超时仍未稳定，使用最后一次测量值
            first_sample_memory = last_memory
            print(f"等待超时，使用当前显存值: {first_sample_memory:.2f} MB")

        print(f"\n第一个样本推理后程序总显存占用: {first_sample_memory:.2f} MB")

        # 等待推理进程完成
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

        # 停止监控
        subprocess.run("pkill -f 'nvidia-smi.*gpu_monitor'", shell=True)

        # 记录结束时间
        end_time = time.time()
        total_time_ms = round((end_time - start_time) * 1000, 2)

        print(stdout)

    except subprocess.CalledProcessError as e:
        print(f"预测失败: {e}")
        print(f"错误输出: {e.stderr}")
        return

    # 获取显存统计
    print("\n" + "=" * 60)
    print("性能统计")
    print("=" * 60)
    print(f"总推理时间: {total_time_ms:.2f} ms")

    # 从nvidia-smi日志读取显存使用
    monitor_log = f"/tmp/gpu_monitor_{os.getpid()}.log"
    peak_memory = 0
    avg_memory = 0
    if os.path.exists(monitor_log):
        with open(monitor_log, 'r') as f:
            memory_samples = [float(line.strip()) for line in f if line.strip()]
        if memory_samples:
            peak_memory_absolute = max(memory_samples)
            avg_memory_absolute = sum(memory_samples) / len(memory_samples)
            # 减去基础显存，得到实际增量
            peak_memory = peak_memory_absolute - initial_memory
            avg_memory = avg_memory_absolute - initial_memory
            first_sample_memory = first_sample_memory - initial_memory
            print(f"\n显存使用统计 (nvidia-smi):")
            print(f"  基础显存: {initial_memory:.2f} MB")
            print(f"  第一个样本后显存增量: {first_sample_memory:.2f} MB")
            print(f"  峰值显存增量: {peak_memory:.2f} MB (绝对值: {peak_memory_absolute:.2f} MB)")
            print(f"  平均显存增量: {avg_memory:.2f} MB (绝对值: {avg_memory_absolute:.2f} MB)")
            print(f"  采样次数: {len(memory_samples)}")
        os.remove(monitor_log)
    else:
        print(f"\n警告: 无法读取显存监控日志")

    # 从 trainer_log.jsonl 读取推理时间并计算统计指标
    avg_time_ms = None
    p50_time_ms = None
    p95_time_ms = None
    p99_time_ms = None
    num_samples = 0

    try:
        import yaml
        import numpy as np

        with open(yaml_config_path, 'r') as f:
            config = yaml.safe_load(f)
        output_dir = config.get('output_dir', '')

        trainer_log_file = Path(output_dir) / 'trainer_log.jsonl'
        if trainer_log_file.exists():
            log_entries = []
            with open(trainer_log_file, 'r') as f:
                for line in f:
                    log_entry = json.loads(line)
                    if 'elapsed_time' in log_entry and 'current_steps' in log_entry:
                        log_entries.append({
                            'elapsed_time': log_entry['elapsed_time'],
                            'current_steps': log_entry['current_steps']
                        })

            if len(log_entries) > 1:
                sample_times = []
                for i in range(1, len(log_entries)):
                    time_diff = log_entries[i]['elapsed_time'] - log_entries[i-1]['elapsed_time']
                    step_diff = log_entries[i]['current_steps'] - log_entries[i-1]['current_steps']

                    if step_diff > 0 and time_diff > 0:
                        if step_diff == 1:
                            # 精确的单样本时间
                            sample_times.append(time_diff)
                        else:
                            # 多个样本的平均时间（近似）
                            avg_time_per_step = time_diff / step_diff
                            for _ in range(step_diff):
                                sample_times.append(avg_time_per_step)

                if sample_times:
                    avg_time_ms = round(np.mean(sample_times), 2)
                    p50_time_ms = round(np.percentile(sample_times, 50), 2)
                    p95_time_ms = round(np.percentile(sample_times, 95), 2)
                    p99_time_ms = round(np.percentile(sample_times, 99), 2)
                    num_samples = len(sample_times)

                    print(f"\n样本推理时间统计 (从 trainer_log.jsonl):")
                    print(f"  总样本数: {num_samples}")
                    print(f"  平均推理时间: {avg_time_ms:.2f} ms")
                    print(f"  P50 (中位数): {p50_time_ms:.2f} ms")
                    print(f"  P95: {p95_time_ms:.2f} ms")
                    print(f"  P99: {p99_time_ms:.2f} ms")
                else:
                    print(f"\n警告: 无法从 trainer_log.jsonl 提取有效的样本时间")
        else:
            print(f"\n警告: 未找到 trainer_log.jsonl 文件: {trainer_log_file}")
    except Exception as e:
        print(f"\n无法计算样本推理时间统计: {e}")

    print("=" * 60)

    # 保存性能指标到JSON
    metrics = {
        'total_time_ms': total_time_ms,
        'initial_memory_mb': initial_memory,
        'first_sample_total_memory_mb': first_sample_memory,
        'peak_memory_mb': peak_memory,
        'avg_memory_mb': avg_memory,
        'device': f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}"
    }

    # 添加样本推理时间统计（如果可用）
    if avg_time_ms is not None:
        metrics['num_samples'] = num_samples
        metrics['avg_sample_time_ms'] = avg_time_ms
        metrics['p50_sample_time_ms'] = p50_time_ms
        metrics['p95_sample_time_ms'] = p95_time_ms
        metrics['p99_sample_time_ms'] = p99_time_ms

    metrics_file = Path(yaml_config_path).parent / 'performance_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n性能指标已保存到: {metrics_file}")


if __name__ == '__main__':
    args = parse_args()
    yaml_path = args.config
    device_id = args.device
    batch_size = args.batch_size

    measure_prediction_performance(yaml_path)

    #python predict_with_metrics.py ner_predict.yaml 0
    
# ============================================================
# 性能统计
# ============================================================
# 总推理时间: 363079.21 ms

# 显存使用统计 (nvidia-smi):
#   基础显存: 10.00 MB
#   第一个样本后显存增量: 1462.00 MB
#   峰值显存增量: 15878.00 MB (绝对值: 15888.00 MB)
#   平均显存增量: 8206.10 MB (绝对值: 8216.10 MB)
#   采样次数: 363

# 样本推理时间统计 (从 trainer_log.jsonl):
#   总样本数: 999
#   平均推理时间: 341.80 ms
#   P50 (中位数): 288.34 ms
#   P95: 802.18 ms
#   P99: 1149.02 ms
# ============================================================


