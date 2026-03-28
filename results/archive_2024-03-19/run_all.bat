@echo off
chcp 65001 >nul
REM IQD实验一键运行脚本 (Windows 11)
REM 用法: run_all.bat [0|1|2|all] [子实验]
REM 示例:
REM   run_all.bat 0        只跑第零层
REM   run_all.bat 1        只跑第一层
REM   run_all.bat 2        只跑第二层（需要GPU）
REM   run_all.bat 2 2a     只跑第二层的实验2a
REM   run_all.bat all      全部跑（默认）

set LAYER=%1
if "%LAYER%"=="" set LAYER=all

set SUB_EXP=%2
if "%SUB_EXP%"=="" set SUB_EXP=all

set SCRIPT_DIR=%~dp0

echo =========================================
echo   IQD 验证实验
echo   层级: %LAYER%
echo =========================================

REM 创建结果目录
if not exist "%SCRIPT_DIR%results\exp0" mkdir "%SCRIPT_DIR%results\exp0"
if not exist "%SCRIPT_DIR%results\exp1" mkdir "%SCRIPT_DIR%results\exp1"
if not exist "%SCRIPT_DIR%results\exp2" mkdir "%SCRIPT_DIR%results\exp2"

cd "%SCRIPT_DIR%experiments"

REM ========== 第零层 ==========
if "%LAYER%"=="all" goto RUN_EXP0
if "%LAYER%"=="0" goto RUN_EXP0
goto SKIP_EXP0

:RUN_EXP0
echo.
echo ^>^>^> 第零层：可控数学函数环境 (CPU)
echo =========================================
python exp0_toy_function.py
if %errorlevel% neq 0 (
    echo [错误] 第零层实验失败
    pause
    exit /b 1
)
echo.
echo ^>^>^> 第零层完成! 检查 results\exp0\
echo =========================================
:SKIP_EXP0

REM ========== 第一层 ==========
if "%LAYER%"=="all" goto RUN_EXP1
if "%LAYER%"=="1" goto RUN_EXP1
goto SKIP_EXP1

:RUN_EXP1
echo.
echo ^>^>^> 第一层：线性回归 + 高斯分布 (CPU)
echo =========================================
python exp1_linear_regression.py
if %errorlevel% neq 0 (
    echo [错误] 第一层实验失败
    pause
    exit /b 1
)
echo.
echo ^>^>^> 第一层完成! 检查 results\exp1\
echo =========================================
:SKIP_EXP1

REM ========== 第二层 ==========
if "%LAYER%"=="all" goto RUN_EXP2
if "%LAYER%"=="2" goto RUN_EXP2
goto SKIP_EXP2

:RUN_EXP2
echo.
echo ^>^>^> 第二层：LLM多代崩溃 (GPU)
echo =========================================

REM 检查GPU
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>nul
if %errorlevel% neq 0 (
    echo [错误] 未检测到可用GPU，跳过第二层
    pause
    exit /b 1
)

python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB')"

python exp2_llm_collapse.py --exp %SUB_EXP%
if %errorlevel% neq 0 (
    echo [错误] 第二层实验失败
    pause
    exit /b 1
)
echo.
echo ^>^>^> 第二层完成! 检查 results\exp2\
echo =========================================
:SKIP_EXP2

echo.
echo =========================================
echo   所有实验完成!
echo   结果目录: %SCRIPT_DIR%results\
echo =========================================
pause
