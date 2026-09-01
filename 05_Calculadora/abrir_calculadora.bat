@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  Calculadora de discinesia (PPMI) - Total-6
echo ============================================
echo.
echo [1/2] Instalando dependencias (so na 1a vez, pode demorar)...
python -m pip install -r requirements.txt
echo.
echo [2/2] Abrindo a calculadora no navegador...
echo (Para fechar depois, volte aqui e aperte Ctrl+C)
python -m streamlit run app.py
pause
