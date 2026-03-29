#!/bin/bash
# 打包实验代码，生成可上传到服务器的 tar.gz
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PACK_NAME="iqd-experiment"
OUTPUT="$PROJECT_DIR/${PACK_NAME}.tar.gz"

echo "========================================="
echo "  打包 IQD 实验代码"
echo "========================================="

cd "$PROJECT_DIR"

tar czf "$OUTPUT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    src/ \
    scripts/ \
    requirements.txt \
    README.md

echo ""
echo ">>> 打包完成: $OUTPUT"
echo ">>> 大小: $(du -h "$OUTPUT" | cut -f1)"
echo ""
echo "包含内容："
tar tzf "$OUTPUT" | head -30
TOTAL=$(tar tzf "$OUTPUT" | wc -l | tr -d ' ')
if [[ "$TOTAL" -gt 30 ]]; then
    echo "  ... (共 $TOTAL 个文件)"
fi
echo ""
echo "========================================="
echo "  上传到服务器后执行："
echo "    tar xzf ${PACK_NAME}.tar.gz"
echo "    bash scripts/setup_env.sh"
echo "    bash scripts/run_all.sh"
echo "========================================="
