Julianna Dietrich
HW3 Ray Casting 

To run project...
Compile: g++ -std=c++17 -O2 main.cc -o hw3
Run: ./hw3 > image.ppm
View: open image.ppm


Before running, ensure you have Python 3 and the Pillow library installed.
Install Pillow via pip if not already installed: "pip install Pillow"
"cd" to the project directory and execute the following command: "python3 main.py"

From there, you will see a progress indicator in the terminal (this may take a tiny bit to fully render due to Anti-aliasing)
    - To remove anti-aliasing for faster render, set SAMPLES_PER_PIXEL to 1 in main.py
Once completed, an image window will pop up displaying the rendered scene. 
The final rendered image will also be saved as "Hw3_RESULT.png" in the project directory.