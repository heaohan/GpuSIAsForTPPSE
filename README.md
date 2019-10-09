# GpuSIAsForTPPSE
This repository includes code and data for SIAs assisted TPPSE.

/SIAs_c++ includes the C++ code of the SIAs. The code is compiled by Visual C++ 2017 and CUDA 10.0. There is a solution: SIAs_c++.sln in this folder. 7 projects are included in the solution: PSO, APSO, DE, LFA, GPU_APSO, GPU_DE, GPU_LFA.

There are some requirements for compiling the code:
1. matlab.props should be loaded to each project;
2. MATLAB should be installed in the computer;
3. add MATLABROOT as the variable of the Environment Variables;
4. add %MATLABROOT%\bin\win64 to the path of the Environment Variables;
5. for GPU_APSO, GPU_DE and GPU_LFA, curand.lib should be added to the linker file.

The compiled .exe files have been listed in the folder x64/Release or  /SIAs_c++/generated_exe. Copy the for_c.mat to either folder, edit the run20.ps1 to chose the .exe to run. When run run20.ps1, certain .exe will run for 20 times, and folders 1 to 20 will be generated, which are the 20 runs results by a certain .exe.
