# cudlb - Pracitce CUDA C++ library

The main purpose of this project is to create a C++ library, which gives users access to Standard Library-like containers and algorithms, in CUDA device runtime code. There are already a number of libraries that do this, the most well known (and complete) is Nvidia's Thrust. This is primarily a pracitce exercise, in which I'll be trying to figure out how the C++ Standard Library containers can be implemented in CUDA runtime. The sources I'll be using during the project include: 
     
     1. Programming: Principles and Pracice Using C++ (2nd Edition) by Bjarne Stroustrup 
     
     2. The C++ Programming Language (4th Edition) by Bjarne Stroustrup 
     
     3. The C++ Standard Library: A Tutorial and Reference (2nd Edition) by Nicolai M. Josuttis 
     
     4. The Modern C++ Challenge by Marius Bancila
    
     5. CUDA C Programming Guide by Nvidia

The Nvidia team has done an excellent job at integrating modern C++ features into CUDA, with a large amount of C++11 and C++14 features now being supported at runtime. 

# How to participate?
Everyone is welcome to join this project, so if you'd like to participate have a look at the TODO list and submit a pull request. 

Please note that this project supports only Windows 10 and is compiled with Microsoft Visual Studio 2017.

If you don't have the CUDA Toolkit installed on your computer, start with:

     1. The first thing you need to do is make sure that your computer is equiped with a supported CUDA GPU:
     https://developer.nvidia.com/cuda-gpus

     2. Download and install the latest Nvidia CUDA Toolkit
     https://developer.nvidia.com/cuda-downloads
    
     3. Once installed, create a new CUDA runtime project in MS Visual Studio 2017. 
    
     4. Rebuild the sample code that comes with each CUDA runtime project.  
    
     5. Add all cudlb files to your project. 
     
     6. Contribute! 
    
 # It is never that easy though... 
 If you're one of those unfortunate folk, who (like me) were stuck at point 2 above and then at point 4, see the section below: 
 
    1. CUDA installation fails 
The latest CUDA Toolkit (9.2 as of writing) still uses older MS Visual Studio 2017 toolkit, so if you have the latest version of MSVS 2017, it is very likely that the installation failed, because the CUDA installer could not integrate nvcc (Nvidia's C++ compiler for CUDA) with Visual Studio. To solve this, you need to install MS Visual Studio 2017 140v Toolset:
https://blogs.msdn.microsoft.com/vcblog/2017/11/02/visual-studio-build-tools-now-include-the-vs2017-and-vs2015-msvc-toolsets/
If you're still having issues installing CUDA, it is likely that your GPUs drivers are a newer version than the one included with the CUDA installer. To fix that you need to roll back your Nvidia GPU drivers to factory defaults and try again. This one can be a pain: 
https://devtalk.nvidia.com/default/topic/1033111/cuda-setup-and-installation/cuda-9-1-cannot-install-due-to-failed-visual-studio-integration/
If the problem persists, then only Google (or God) can help you. 

    2. I got this thing installed. Sample code won't compile. 
So you have the CUDA Toolkit installed, but when you create a new CUDA Runtime project in MSVS 2017, it fails to compile. The error you are likely getting is:
 
   Error C1189 #error: -- unsupported Microsoft Visual Studio version! 

Even though you have MSVS v140 Toolset installed, by default MSVS 2017 compiles new projects using v141 Toolset. To fix that you'll have to tell the compiler that you're using the older toolset:
https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017/

If you got this far, you should be able to build the sample code. Once that's done you're all set up and ready to go! 

  




