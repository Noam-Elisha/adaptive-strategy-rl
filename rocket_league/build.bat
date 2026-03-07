@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set PATH=%~dp0ninja_bin;%PATH%

echo ===CONFIGURE_START===
cmake -B "%~dp0build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\nvcc.exe" -DCMAKE_CUDA_FLAGS="--allow-unsupported-compiler"
echo ===CONFIGURE_RC=%ERRORLEVEL%===

if %ERRORLEVEL% NEQ 0 exit /b 1

echo ===BUILD_START===
cmake --build "%~dp0build" --config RelWithDebInfo -j 8
echo ===BUILD_RC=%ERRORLEVEL%===
