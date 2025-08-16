@echo off
echo ========================================
echo Building Enhanced Brainstorm DLL
echo ========================================

cd ImmolateCPP

REM Check if CMake is installed
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo CMake not found. Please install CMake first.
    echo You can install it with: winget install --id Kitware.CMake
    pause
    exit /b 1
)

REM Check for Visual Studio Build Tools
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Visual Studio Build Tools not found.
    echo Please run this from a Visual Studio Developer Command Prompt
    echo or install Visual Studio Build Tools first.
    pause
    exit /b 1
)

REM Clean previous build
if exist build_brainstorm (
    echo Cleaning previous build...
    rmdir /s /q build_brainstorm
)

REM Generate build files
echo Generating build files...
cmake -G "Visual Studio 17 2022" -A x64 -B build_brainstorm -DCMAKE_BUILD_TYPE=Release CMakeLists_Brainstorm.txt
if %errorlevel% neq 0 (
    echo CMake configuration failed.
    pause
    exit /b 1
)

REM Build the DLL
echo Building Brainstorm DLL...
cmake --build build_brainstorm --config Release
if %errorlevel% neq 0 (
    echo Build failed.
    pause
    exit /b 1
)

REM Copy the DLL to the mod folder
echo Copying DLL...
copy build_brainstorm\Release\Immolate.dll ..\Immolate_new.dll
if %errorlevel% neq 0 (
    copy build_brainstorm\Immolate.dll ..\Immolate_new.dll
)

cd ..

echo.
echo ========================================
echo Build Complete!
echo ========================================
echo.
echo New DLL has been built as Immolate_new.dll
echo.
echo To use it, rename the old Immolate.dll to Immolate_old.dll
echo and rename Immolate_new.dll to Immolate.dll
echo.
pause