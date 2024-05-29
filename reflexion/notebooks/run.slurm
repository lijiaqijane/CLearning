#!/bin/bash
#SBATCH -J ram                  # 作业名为 test
#SBATCH -o test.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -p DGX                    # 作业提交的分区为 HGX
#SBATCH --qos=lv0b               # 作业使用的 QoS 为 lv0b
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH -t 10:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡

# 输入要执行的命令，例如 ./hello 或 python test.py 等
eval "$(/home/zhangxiaoyuan/miniconda3/bin/conda shell.bash hook)"
conda activate craft
module load cuda11.7/toolkit/11.7.1

python Ours.py

