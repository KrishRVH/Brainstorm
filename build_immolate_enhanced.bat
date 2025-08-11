@echo off
echo ========================================
echo Building Enhanced Immolate with Erratic Deck Support
echo ========================================

cd ImmolateSourceCode

REM Check if CMake is installed
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo CMake not found. Installing with winget...
    winget install --id Kitware.CMake
    if %errorlevel% neq 0 (
        echo Failed to install CMake. Please install manually.
        pause
        exit /b 1
    )
)

REM Check for Visual Studio Build Tools
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Visual Studio Build Tools not found. Installing...
    winget install Microsoft.VisualStudio.2022.BuildTools --force --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22000"
    if %errorlevel% neq 0 (
        echo Failed to install Build Tools. Please install manually.
        pause
        exit /b 1
    )
)

REM Generate build files
echo Generating build files...
cmake -G "Visual Studio 17 2022" -A x64 -B build
if %errorlevel% neq 0 (
    echo CMake configuration failed.
    pause
    exit /b 1
)

REM Build the project
echo Building Immolate...
cmake --build build --config Release
if %errorlevel% neq 0 (
    echo Build failed.
    pause
    exit /b 1
)

REM Copy the executable
echo Copying executable...
copy build\Release\Immolate.exe ..\Immolate_Enhanced.exe

cd ..

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo Enhanced Immolate has been built as Immolate_Enhanced.exe
echo.
echo To use the Erratic deck filter, run:
echo   Immolate_Enhanced.exe -f erratic_deck_enhanced
echo.
echo To find glitched seeds like 7LB2WVPK:
echo   Immolate_Enhanced.exe -f erratic_deck_enhanced -s 7LB2WVPK -n 1
echo.
echo For high face count + suit ratio:
echo   Immolate_Enhanced.exe -f erratic_deck_enhanced -c 2000
echo.
pause