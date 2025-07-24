@echo off
REM =================================================================================
REM ExportTree.bat
REM Экспортирует структуру каталога (включая файлы) в текстовый файл.
REM 
REM Использование:
REM   ExportTree.bat [путь_к_корню] [имя_выходного_файла]
REM Параметры по умолчанию:
REM   путь_к_корню     = .  (текущая папка)
REM   имя_выходного_файла = directory_tree.txt
REM =================================================================================

SETLOCAL ENABLEDELAYEDEXPANSION

REM 1) Определяем корневую папку
IF "%~1"=="" (
    SET "ROOT_DIR=."
) ELSE (
    SET "ROOT_DIR=%~1"
)

REM 2) Определяем выходной файл
IF "%~2"=="" (
    SET "OUTPUT_FILE=directory_tree.txt"
) ELSE (
    SET "OUTPUT_FILE=%~2"
)

REM 3) Генерируем дерево
tree "%ROOT_DIR%" /F /A > "%OUTPUT_FILE%"

REM 4) Сообщаем результат
ECHO Структура каталогов для "%ROOT_DIR%" сохранена в "%OUTPUT_FILE%".

ENDLOCAL
