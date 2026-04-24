@echo off
setlocal
set ROOT=%~dp0

echo =^> Installing backend dependencies...
pip install -r "%ROOT%backend\requirements.txt"

echo =^> Installing frontend dependencies...
pushd "%ROOT%frontend"
npm install
popd

echo.
echo =^> Starting backend  -^> http://localhost:8000
echo =^> Starting frontend -^> http://localhost:5173
echo.

start "RAG Backend"  cmd /k "cd /d "%ROOT%backend" && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
start "RAG Frontend" cmd /k "cd /d "%ROOT%frontend" && npm run dev -- --host 0.0.0.0"
