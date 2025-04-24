### To Run Code

1. Go to 'build' directory
2. if you have chaged code so please remove existing files under build directory using 'rm -r *'
3. cmake -G "MinGW Makefiles" ..
4. cmake --build . --clean-first
5. Run .exe file
    - .\bin\parallel_bfs.exe (Default)
    - .\bin\parallel_bfs.exe 10000 0.01 (Random Synthetic Graphs)
    - .\bin\parallel_bfs.exe ..\graphs\WikiTalk.txt (To Work with Real World Data)