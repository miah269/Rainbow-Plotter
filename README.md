# Rainbow-Plotter

Congratulations on acquiring the world's one and only Rainbow Plotter by MAD.

To use this machine, download the following:
1. SVGtoGcode app from this repository
   a. For macOS: SVGtoGcode
   b. For Windows: SVGtoGcode.exe
3. GRBL-Plotter: https://grbl-plotter.de/

Before running the machine:
1. Load six Parrot Products Slimline markers into their designated slots in the toolhead
2. Plug the machine into a wall socket
3. Disengage the emergency stop
4. Connect your computer to the machine's Arduino via USB cable

Then follow the steps in this video to convert an image into G-code and run a job:

https://github.com/user-attachments/assets/0d8db67c-4aea-4ee3-9e06-1e359e685a9c


The source code for the SVGtoGCode program can be found in the SVGtoGCode folder in this repository. This program converts SVG images to GCode instructions for the Rainbow Plotter to execute. Credit is due to arcadeperfect for his svg2gcode_grbl repository (https://github.com/arcadeperfect/svg2gcode_grbl), which was adapted to create the program used for this project.

A full technical report (MHam_Skripsiev5.pdf) is also included in this repository, detailing the design, build, software development and testing process of the Rainbow Plotter.
