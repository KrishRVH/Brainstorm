@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo GPU Detection and Performance Test
echo ========================================
echo.

REM Navigate to project root
cd /d "%~dp0\..\.."

REM Check for Immolate
set IMMOLATE=
if exist tools\Immolate.exe (
    set IMMOLATE=tools\Immolate.exe
) else if exist ImmolateSourceCode\build\Release\Immolate.exe (
    set IMMOLATE=ImmolateSourceCode\build\Release\Immolate.exe
) else (
    echo ERROR: Immolate.exe not found
    echo Please build first: scripts\build\build_windows.bat
    exit /b 1
)

echo Using: %IMMOLATE%
echo.

REM Test 1: List OpenCL devices
echo OpenCL Device Detection:
echo ========================
%IMMOLATE% --list_devices

if %errorlevel% neq 0 (
    echo.
    echo WARNING: No OpenCL devices detected
    echo.
    echo Possible solutions:
    echo 1. Install latest GPU drivers
    echo 2. For NVIDIA: Install CUDA Toolkit
    echo 3. For AMD: Install AMD APP SDK
    echo 4. For Intel: Install Intel OpenCL Runtime
    echo.
    pause
    exit /b 1
)

echo.
echo Performance Benchmarks:
echo =======================
echo.

REM Benchmark function
echo Running benchmarks...
echo.

REM Basic test
echo Test: Basic filter [10,000 seeds]
powershell -Command "$start = Get-Date; & '%IMMOLATE%' -f ImmolateSourceCode\filters\test -n 10000 | Out-Null; $elapsed = (Get-Date) - $start; Write-Host \"Time: $($elapsed.TotalSeconds)s\"; Write-Host \"Rate: $([math]::Round(10000 / $elapsed.TotalSeconds)) seeds/second\""
echo.

REM Erratic test
if exist ImmolateSourceCode\filters\erratic_brainstorm.cl (
    echo Test: Erratic deck filter [10,000 seeds]
    powershell -Command "$start = Get-Date; & '%IMMOLATE%' -f ImmolateSourceCode\filters\erratic_brainstorm -n 10000 | Out-Null; $elapsed = (Get-Date) - $start; Write-Host \"Time: $($elapsed.TotalSeconds)s\"; Write-Host \"Rate: $([math]::Round(10000 / $elapsed.TotalSeconds)) seeds/second\""
    echo.
)

REM Large test
echo Test: Large-scale [100,000 seeds]
powershell -Command "$start = Get-Date; & '%IMMOLATE%' -f ImmolateSourceCode\filters\test -n 100000 | Out-Null; $elapsed = (Get-Date) - $start; Write-Host \"Time: $($elapsed.TotalSeconds)s\"; Write-Host \"Rate: $([math]::Round(100000 / $elapsed.TotalSeconds)) seeds/second\""
echo.

echo ========================================
echo Performance Analysis
echo ========================================
echo.
echo Expected Performance:
echo - CPU-only: 1,000-5,000 seeds/second
echo - Integrated GPU: 10,000-50,000 seeds/second  
echo - Dedicated GPU: 100,000+ seeds/second
echo.
echo Your performance depends on:
echo - GPU model and drivers
echo - OpenCL implementation
echo - Filter complexity
echo - System memory bandwidth
echo.

pause