@echo off
echo ========================================
echo Testing Immolate with Erratic Deck Support  
echo ========================================
echo.

cd ImmolateSourceCode

if not exist Immolate.exe (
    echo ERROR: Immolate.exe not found!
    echo Please run build_immolate_enhanced.bat first.
    pause
    exit /b 1
)

echo Testing known glitched seed 7LB2WVPK...
echo This should produce 52 copies of 10 of Spades
echo.
Immolate.exe -f erratic_brainstorm -s 7LB2WVPK -n 1 -c 1
echo.

echo ----------------------------------------
echo Testing random seeds for high face cards...
echo Looking for 20+ face cards with 50%+ suit ratio
echo.
Immolate.exe -f erratic_brainstorm -s random -n 10000 -c 2000
echo.

echo ----------------------------------------
echo Testing specific seed range...
echo.
Immolate.exe -f erratic_brainstorm -s AAAAAAAA -n 1000 -c 2500

cd ..

echo.
echo ========================================
echo Test Complete!
echo ========================================
echo.
echo If you found interesting seeds, you can test them in Balatro
echo to verify our Erratic deck generation is accurate.
echo.
echo Remember: The original Immolate uses GPU acceleration,
echo so it will be MUCH faster than our CPU implementation.
echo.
pause