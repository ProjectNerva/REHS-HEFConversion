# REHS-AIHATConversion

Convert an Ultralytics YOLO model(.pt) into a hailo .hef file, for deployment on Hailo-8/Hailo-8l accelerators

## Prerequites
- Docker needs to be downloaded + an account is needed, too. Within settings, if you are on a macOS, enable "Use Rosetta for x86_64/amd64 emulation on Apple Silicon."
- The Hailo Dataflow Compiler is also needed. This repo is attuned to the Python Version 3.10. The details for the download is going to be AI Software Suite, Dataflow Compiler, x86, Linux, and 3.10 and go to "Filter by" and set it to "Archive" and download version 3.33.1
- The pt file you want to convert

## General Workflow
1. Create the docker container
```
docker build -t hef-conversion .
```
2. Create a python virtual environment and install all the packages in `requirements.txt`
3. Execute the `main.py` file in order to convert the pt file into an onnx file.
```
python main.py (pt model)
```
4. Create a `shared_data directory` and move the onnx file in there.
5. Execute `build_calib_set.py` in order to build a calibration file and move it to the `shared_data`. Note that the arguments are the directory of the images and the output name.
```
python build_calib_set.py (input directory) (output .npy file)
```
6. Run the docker
```
docker run -it -v "$(pwd)/shared_data:/app/shared_data" hef-conversion
```
7. `cd` into `shared_data` and execute `main.py`
```
python3 hef_converter.py (input onnx file without .onnx) (har file name) (calibration set without .npy) (output hef file name)
```
