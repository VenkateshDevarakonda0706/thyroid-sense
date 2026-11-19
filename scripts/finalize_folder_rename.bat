@echo off
REM ==============================================================================
REM ThyroidSense - Folder Rename Finalization Helper Script
REM ==============================================================================
REM Run this script only if you want to permanently rename the folder on disk
REM from "Thyroid complete" to "ThyroidSense" (after closing any IDE processes
REM that have handles locked on "Thyroid complete").
REM
REM Note: The directory junction "ThyroidSense" already allows opening and working
REM in C:\Users\Venkatesh\OneDrive\Desktop\Projects\ThyroidSense without running this.
REM ==============================================================================

cd /d "C:\Users\Venkatesh\OneDrive\Desktop\Projects"

echo Checking for directory junction...
if exist "ThyroidSense" (
    echo Removing directory junction ThyroidSense...
    rmdir "ThyroidSense"
)

echo Renaming "Thyroid complete" to "ThyroidSense"...
ren "Thyroid complete" "ThyroidSense"

if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Successfully renamed "Thyroid complete" to "ThyroidSense"!
    echo You can now open C:\Users\Venkatesh\OneDrive\Desktop\Projects\ThyroidSense in Antigravity or VS Code.
) else (
    echo [WARNING] Could not rename folder. A process (e.g. IDE or terminal) is still locking the directory.
    echo Re-creating directory junction to ensure continuous access...
    mklink /J "ThyroidSense" "Thyroid complete"
    echo Junction restored.
)
pause
