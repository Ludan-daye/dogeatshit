@echo off
chcp 65001 >nul
REM IQD实验环境搭建 — Windows 11 + RTX 5080 16GB
echo =========================================
echo   IQD 实验环境搭建
echo   GPU: RTX 5080 16GB / Windows 11
echo =========================================

REM 1. 检查conda
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 请先安装 Miniconda/Anaconda
    echo   https://docs.anaconda.com/miniconda/install/#quick-command-line-install
    exit /b 1
)

REM 2. 创建conda环境
echo [1/4] 创建conda环境 iqd (Python 3.10)...
call conda create -n iqd python=3.10 -y
call conda activate iqd

REM 3. 安装PyTorch (CUDA 12.x)
echo [2/4] 安装PyTorch (CUDA 12.x)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM 4. 安装其他依赖
echo [3/4] 安装项目依赖...
pip install -r requirements.txt

REM 5. 验证
echo [4/4] 验证安装...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU'); import mauve; print('MAUVE: OK'); import transformers; print(f'Transformers: {transformers.__version__}'); print('所有依赖安装成功!')"

REM 创建结果目录
if not exist results\exp0 mkdir results\exp0
if not exist results\exp1 mkdir results\exp1
if not exist results\exp2 mkdir results\exp2

echo =========================================
echo   环境搭建完成!
echo   激活环境: conda activate iqd
echo   运行实验: run_all.bat
echo =========================================
pause
