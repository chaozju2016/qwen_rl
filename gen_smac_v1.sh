#!/bin/bash

# SMAC评估自动化脚本
# 用于批量执行不同地图的模型评估

# 设置基础路径
OBS_TO_TEXT_SCRIPT="/home/wangchao/work/qwen_rl/obs_to_text_v1.py"

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
DATASET_DIR="/mnt/HDD/wangchao/smac_v1_json"
HF_REPO="chaozju2016/smac_v1_json"

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