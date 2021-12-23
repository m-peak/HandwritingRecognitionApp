
-----------------------
Handwriting Recognition
-----------------------

A standalone executable application for Handwriting Recognition on the UI.


Requirements
------------

* Use the main moniter if you are using multiple monitors.

* For Mac, most display resolutions are supported, except Macbook Pro.

* For Windows, use the display scale: 100%.


Application Execution
---------------------

 * Please download the application from the following links.

   Mac:
   'app_mac.zip' : https://www.dropbox.com/s/1o0pvq15udu0a60/app_mac.zip?dl=0

   - Unzip the file and double-click 'app_mac' (the application will not be allowed to execute for the first time).
   - Right-click the 'app_mac' to 'Open', or 'Open With' and choose 'Terminal'. It may take around 10 seconds to start up.
   - Please place the UI at the top-left corner of your screen in case the application does not seem working correctly
     (due to the fact that some Mac environment might prevent the application from capturing the UI).

   Windows:
   'app_win.zip' : https://www.dropbox.com/s/kne7lvduhljvzmg/app_win.zip?dl=0

   - Unzip the file and double-click 'app_win.exe' (the application will not allowed to execute for the first time).
   - Click 'More info' in the windows' warning popup window and click 'Run anyway'. It may take more than 10 seconds to start up.


Usage
-----

 * Write something in block letters using a mouse, track pad or touch screen.

 * Click the 'Recognize' button to predict the handwritten characters.

 * 'Result' shows the predicted characters, and 'Accuracy' shows the accuracy rate for each character.


File Structure in the 'masamip2_proj_fn_min.zip' uploaded to Cousera
----------------------------------------------------------------

 Python files are mainly inclued in this zip file for the demonstration of the code for the application.

 * README.txt: explaining how to use the application and development environment (this file)

 * app.py: controlling the UI

 * recognizer.py: recognizing the characters via UI

 * modeler.py: setting parameters for a deep learning model and running trainer.py

 * trainer.py: training the dataset for fitting a deep learning model

 * utils.py: sharing some functions in other python class files


The source code and other related files for the development uploaded to Dropbox
---------------------------------------------------------------------------

 All the code and other related files (e.g. models) are available at the following links.

 * model_pad_fill.h5: a deep learning model with data augmentations (rotation, shift, shear and zoom), pad and fill

 * model.h5: a deep learning model with data augmentations (rotation, shift, shear and zoom)

 * English directory: handwriting character dataset (some png files are cleaned)


 Mac:
 'masamip2_proj_fn_mac.zip' : https://www.dropbox.com/s/ch9tbmxeerk91uu/masamip2_proj_fn_mac.zip?dl=0

 - app.spec: a config file for mac to generate an executable 'app' file (some adjustments are required for your environment)
 - app_mac.spec: a backup of 'app.spec' as a reference

 Windows:
 'masamip2_proj_fn_win.zip' : https://www.dropbox.com/s/32tg54hjefa408h/masamip2_proj_fn_win.zip?dl=0

 - app.spec: a config file for windows to generate an executable 'app' file (some adjustments are required for your environment)
 - app_win.spec: a backup of 'app.spec' as a reference


Running the Application Locally
-------------------------------

 * Handwriting Recognition application can be run locally by using the following steps.

   Steps for Running the Application Locally:

   - Set up your development environment (see the section 'Steps for Packaging' below).

   - The following command line can be run if python3.7 is set up to run by a command line 'python3.7'.

     python3.7 app.py


Fitting a Deep Learning Model
-----------------------------

 * Deep learning models can be fitted by using the following steps.

   Steps for Fitting Deep Learning Models:

   - Set up your development environment (see the section 'Steps for Packaging' below).

     Running modeler.py which takes a parameter of a model file name.
     A file name with '_pad' will include the data augmentation of padding and '_fill' will include the data augmentation of filling.

   - The following command lines can be run if python3.7 is set up to run by a command line 'python3.7'.

     python3.7 modeler.py model_pad_fill.h5
     python3.7 modeler.py model.h5


Packaging the Code
------------------

 * The code can be packaged by using the following steps.

   Steps for Packaging:

   - Set up your development environment.

     This application can be built and packaged on the following environments.

     * Mac: OS 10.14.5 (Mojave)
       - freetype (2.10.1): Software library to render fonts (required for freezing this application with libpng16 by pyinstaller)
       - hdf5 (1.12.0): File format designed to store large amounts of data (required for freezing this application with .h5 file by pyinstaller)

     * Windows: 10 Pro (1909)

     * Python 3.7.7: (tkinter: the standard Python interface to the Tk GUI toolkit)

     * Python Libraries (Dependencies) and their descriptions
       - keras (2.3.0): Deep Learing for humans
       - opencv-contrib-python (4.2.0.34): Wrapper package for OpenCV python bindings
       - pillow (7.1.1): Python Imaging Library (for grabbing an image from a GUI)
       - matplotlib (3.2.1): Python plotting package (for debugging purpose)
       - pyinstaller (3.6): PyInstaller bundles a Python application and all its dependencies into a single package
       - tensorflow (2.0.0): TensorFlow is an open source machine learning framework for everyone

       Mac:
       - pyobjc-framework-Quartz (6.2): Wrappers for the Quartz frameworks on macOS (Quartz for retrieving Mac GUI information)
       - h5py (2.8.0): Read and write HDF5 files from Python (required for freezing this application by pyinstaller)
       - protobuf (3.8.0): Protocol Buffers (required for freezing this application by pyinstaller)
       - setuptools (46.1.3): Easily download, build, install, upgrade, and uninstall Python packages (required for freezing this application by pyinstaller)

       Windows:
       - pywin32 (227): Python for Window Extensions (including win32gui for retrieving Windows GUI information)

   - (Optional) The following command line creates an app.spec file.

     pyi-makespec --onefile app.py

   - Configure the app.spec (modify the 'dir' parameter for 'pathex' entry and other settings if necessary).

   - The following command line creates an executable file 'app' (for Mac) or 'app.exe' (for Windows) under dist directory which will be located at the same directory level as app.py.

     pyinstaller --onefile app.spec


