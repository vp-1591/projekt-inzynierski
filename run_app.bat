@echo off
echo Starting AntyDezinformator Services...

start cmd /k "cd backend && python -m app.main"
start cmd /k "cd frontend && npm run dev"

echo Services are starting in separate windows.
