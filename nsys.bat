@echo off
::This script exists to help you use the NSight Systems profiler on managed desktops
::The GUI cannot be used to launch profiling, as this performs a feature check requiring administrator privileges
::Simply call the batch file in cmd specifying the path to your executable, and any trailing arguments required. It will then output nsys_report.nsys-rep
::You may want to adjust the call arguments, simply avoid passing --sample=system-wide or --cpucyxsw=system-wide
::e.g. nsys.bat "x64\Release\assignment.exe" CUDA SD 12 100000

set nsight-systems="C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe"
IF "%1"=="" (
    echo "Usage: %0 <executable> <run args>"
	pause
	goto :eof
)
%nsight-systems% profile --trace=cuda --sample=process-tree,nvtx --cpuctxsw=process-tree -o nsys_report %*