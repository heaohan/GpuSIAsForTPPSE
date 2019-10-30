# GpuSIAsForTPRPSE
This repository includes code and data for SIAs assisted TPRPSE.

/SIAs_c++ includes the C++ code of the SIAs. The code is compiled by Visual C++ 2017 and CUDA 10.0. There is a solution: SIAs_c++.sln in this folder. 7 projects are included in the solution: PSO, APSO, DE, LFA, GPU_APSO, GPU_DE, GPU_LFA.

There are some requirements for compiling the code:
1. matlab.props should be loaded to each project;
2. MATLAB should be installed in the computer;
3. add MATLABROOT as the variable of the Environment Variables;
4. add %MATLABROOT%\bin\win64 to the path of the Environment Variables;
5. for GPU_APSO, GPU_DE and GPU_LFA, curand.lib should be added to the linker file.

To run the code, MATLAB should be installed in the computer.

/simulation/sample_generation stores the code and data of the simulated interferograms. (including for_c.mat)
/experiment/sample_generation stores the data and images of the experimental interferograms. (including for_c.mat)

The compiled .exe files have been listed in the folder x64/Release or /SIAs_c++/generated_exe. Copy the for_c.mat to either folder, edit the run20.ps1 to chose the .exe to run. When run run20.ps1, certain .exe will run for 20 times, and folders 1 to 20 will be generated, which are the 20 runs results by a certain .exe.

/results show the results with the interferograms. Run furtherProcess.m or furtherProcessBatch.m in the /simulation/post_processing or /experiment/post_processing, for the simulation results (/results/sim*) or experiment results (/results/exp*), respectively. For furtherProcess.m, when the folder selection dialog window pops up after running, select the inner folder, e.g., /results/exp1/APSO/1, while for furtherProcessBatch.m, select the outer folder, e.g., /results/exp1/APSO. For furtherProcess.m in the /experiment/post_processing, if choose the PSI, can chose the phase shifted interferograms in /results/sample_generation, e.g., (meanRec_1_cutted.bmp, meanRec_111_cutted.bmp, meanRec_139_cutted.bmp, meanRec_168_cutted.bmp) in /results/sample_generation/sample1.
