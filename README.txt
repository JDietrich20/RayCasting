Julianna Dietrich
Ray Casting Assignment 

* <= 3 spheres + 1 plane
* Point light source (intensity and location)
* Eye position
* Scene configuration
* Surface properties (k value in lighting model): Diffuse, Ambient, Specular
* View plane size and location 
* Shadow rays 
* Anti-aliasing (supersampling)

Extra:
(X) Texture Mapping 
(X) Add other objects like ellipsoids, triangles, boxes, etc. 
(X) Ray Tracing with Reflections and Refractions
___________________________________________________________________________

Before running, ensure you have Python 3 and the Pillow library installed.
Install Pillow via pip if not already installed: "pip install Pillow"
"cd" to the project directory and execute the following command: "python3 <inser_main_file_name_here>.py"
            - Ray Casting main file: python3 mainRC.py
            - Ray Tracing main file: python3 mainRT,py 
    - Both files are set up to render the same scene, but with different techniques (rendering algorithm)

________________________________________________________________________

From there, you will see a progress indicator in the terminal (this may take a tiny bit to fully render due to Anti-aliasing)
    - To remove anti-aliasing for faster render, set SAMPLES_PER_PIXEL to 1 in main.py

Once completed, an image window will pop up displaying the rendered scene. 

The final rendered image will also be saved as "Hw3_RAYCASTING.png" or "Hw3_RAYTRACING.png" in the project directory depending on the main file ran.