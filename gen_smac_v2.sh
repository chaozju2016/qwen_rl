#!/bin/bash

# SMAC评估自动化脚本
# 用于批量执行不同地图的模型评估

# 设置基础路径
TRAIN_SCRIPT="/home/wangchao/work/marl-ppo-suite/train.py"
MODEL_BASE_PATH="~/model/legalaspro"
OBS_TO_TEXT_SCRIPT="/home/wangchao/work/qwen_rl/obs_to_text.py"

# 定义游戏地图列表
GAMES=(
    "protoss_10_vs_10"
    "protoss_5_vs_5"
    "terran_10_vs_10"
    "terran_5_vs_5"
    "zerg_10_vs_10"
    "zerg_10_vs_11"
    "zerg_5_vs_5"
)

# 让用户判断Algo是mappo还是happo，如果用户直接按回车，默认是mappo
read -p "请输入算法类型 (默认 mappo): " ALGO
if [ -z "$ALGO" ]; then
    ALGO="mappo"
fi
# 如果用户输入的既非mappo也非happo，则提示错误并退出
if [[ "$ALGO" != "mappo" && "$ALGO" != "happo" ]]; then
    echo "❌ 错误的算法类型: $ALGO"
    echo "请使用 'mappo' 或 'happo'。"
    exit 1
fi

# 基础参数
BASE_ARGS="--env_name smacv2 --mode eval --use_rnn --use_value_norm --state_type AS --eval_episodes 2000 --n_eval_rollout_threads 32 --seed 1 --algo $ALGO"

# 输出开始信息
echo "============================================"
echo "开始批量执行SMAC模型评估"
echo "总共需要评估 ${#GAMES[@]} 个地图"
echo "============================================"

# 记录开始时间
START_TIME=$(date +%s)

# 遍历所有游戏进行评估
for i in "${!GAMES[@]}"; do
    GAME="${GAMES[$i]}"
    CURRENT_NUM=$((i + 1))
    
    echo ""
    echo "[$CURRENT_NUM/${#GAMES[@]}] 正在评估地图: $GAME"
    echo "----------------------------------------"
    
    # 构建模型路径
    MODEL_PATH="$MODEL_BASE_PATH/$(echo "$ALGO" | tr '[:lower:]' '[:upper:]')-smacv2_${GAME}-seed1/final-torch.model"
    
    # 构建完整命令
    COMMAND="python $TRAIN_SCRIPT $BASE_ARGS --map_name $GAME --model $MODEL_PATH"
    
    # 显示即将执行的命令
    echo "执行命令: $COMMAND"
    
    # 执行命令
    if eval $COMMAND; then
        echo "✅ $GAME 评估完成"
        # 休眠10秒以避免过快输出
        sleep 10
    else
        echo "❌ $GAME 评估失败，错误码: $?"
        echo "继续执行下一个地图..."
    fi
    
    echo "----------------------------------------"
done

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
MINUTES=$((TOTAL_TIME / 60))
SECONDS=$((TOTAL_TIME % 60))

echo ""
echo "============================================"
echo "所有地图评估完成！"
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo "============================================"

# 执行obs_to_text.py
echo ""
echo "正在执行观测数据转换..."
echo "命令: python $OBS_TO_TEXT_SCRIPT"

if python "$OBS_TO_TEXT_SCRIPT"; then
    echo "✅ 观测数据转换完成"
else
    echo "❌ 观测数据转换失败，错误码: $?"
fi

# 推送数据到Hugging Face
echo ""
echo "============================================"
echo "开始推送数据集到Hugging Face"
echo "============================================"

# 数据集目录
DATASET_DIR="/mnt/HDD/wangchao/smac_v2_json"
HF_REPO="chaozju2016/smac_v2_json"

# 检查目录是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ 数据集目录不存在: $DATASET_DIR"
    exit 1
fi

# 检查huggingface-cli是否可用
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli 未找到，请先安装 huggingface_hub"
    echo "安装命令: pip install huggingface_hub"
    exit 1
fi

echo "数据集目录: $DATASET_DIR"
echo "目标仓库: https://huggingface.co/datasets/$HF_REPO"
echo ""

# 获取目录中的所有文件
FILES=($(find "$DATASET_DIR" -type f -name "*.json*" | sort))

if [ ${#FILES[@]} -eq 0 ]; then
    echo "❌ 在 $DATASET_DIR 中未找到JSON文件"
    exit 1
fi

echo "发现 ${#FILES[@]} 个文件需要上传："
for file in "${FILES[@]}"; do
    echo "  - $(basename "$file")"
done
echo ""

# 记录上传开始时间
UPLOAD_START_TIME=$(date +%s)

# 逐个上传文件
for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILENAME=$(basename "$FILE")
    CURRENT_NUM=$((i + 1))
    
    echo "[$CURRENT_NUM/${#FILES[@]}] 正在上传: $FILENAME"
    echo "文件路径: $FILE"
    
    # 构建上传命令
    UPLOAD_COMMAND="huggingface-cli upload $HF_REPO \"$FILE\" --repo-type dataset"
    
    # 执行上传
    if eval $UPLOAD_COMMAND; then
        echo "✅ $FILENAME 上传成功"
    else
        echo "❌ $FILENAME 上传失败，错误码: $?"
        echo "继续上传下一个文件..."
    fi
    
    echo "----------------------------------------"
done

# 计算上传总耗时
UPLOAD_END_TIME=$(date +%s)
UPLOAD_TOTAL_TIME=$((UPLOAD_END_TIME - UPLOAD_START_TIME))
UPLOAD_MINUTES=$((UPLOAD_TOTAL_TIME / 60))
UPLOAD_SECONDS=$((UPLOAD_TOTAL_TIME % 60))

echo ""
echo "============================================"
echo "数据集上传完成！"
echo "上传耗时: ${UPLOAD_MINUTES}分${UPLOAD_SECONDS}秒"
echo "Hugging Face链接: https://huggingface.co/datasets/$HF_REPO"
echo "============================================"

echo ""
echo "🎉 全部任务执行完成！"