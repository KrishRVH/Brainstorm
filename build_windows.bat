@echo off
echo Building Immolate.dll for Windows...

REM Check for Visual Studio compiler
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Visual Studio compiler not found. Trying MinGW...
    goto mingw
)

:msvc
echo Using MSVC compiler...
cl /O2 /LD /Fe:immolate.dll immolate.c
if %errorlevel% equ 0 (
    echo Build successful with MSVC!
    goto end
) else (
    echo MSVC build failed. Trying MinGW...
    goto mingw
)

:mingw
where gcc >nul 2>nul
if %errorlevel% neq 0 (
    echo MinGW gcc not found. Please install MinGW or Visual Studio.
    goto error
)

echo Using MinGW compiler...
gcc -O3 -shared -o immolate.dll immolate.c -lm
if %errorlevel% equ 0 (
    echo Build successful with MinGW!
    goto end
) else (
    echo MinGW build failed.
    goto error
)

:error
echo.
echo Build failed. Please ensure you have either:
echo 1. Visual Studio with C++ tools installed
echo 2. MinGW with gcc installed
pause
exit /b 1

:end
echo.
echo Immolate.dll created successfully!
echo You can now use it with the Brainstorm mod.
pause