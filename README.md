## Welcome to pydrodem!
**pydrodem** is a collection of python-based tools for hydrologically conditioning digital elevation models (DEMs). The DEM conditioning tools are designed to wrap around the functionality of Geopandas, WhiteBoxTools, and TauDEM to allow for high resolution DEMs to be conditioned in a flexible, user-customizable way. 

The codes were developed for a parallel computing environment, but can also be run on a single machine or within the cloud by running the appropriate shell script.

New versions are still in development. If you find a bug or issue, please let the team know by submitting a ticket.

### Origin
pydrodem was developed at the National Geospatial-Intelligence Agency (NGA) by federal government employees in the course of their official duties, so it is <strong>not</strong> subject to copyright protection and is in the public domain in the United States. 

You are free to use the core public domain portions of pydrodem for any purpose. Modifications back to the cores of any dependency functions are subject to the original licenses and are separate from the core public domain work of FDE. 

### Transparency
NGA is posting code created by government officers in their official duties in transparent platforms to increase the impact and reach of taxpayer-funded code. NGA is also posting pydrodem to give NGA partners insight into how elevation data is quality controlled within NGA to help partners understand parts of quality assurance checks.

### Pull Requests
If you'd like to contribute to this project, please make a pull request. We'll review the pull request and discuss the changes. This project is in the public domain within the United States and all changes to the core public domain portions will be released back into the public domain. By submitting a pull request, you are agreeing to comply with this waiver of copyright interest. Modifications to dependencies under copyright-based open source licenses are subject to the original license conditions.

### Dependencies
Whitebox Tools

TauDEM

Boto3

Botocore

GDAL

Geopandas

Numpy

Scipy

Shapely

Pandas

Matplotlib

### Quick Start

1. Pull or download the pydrodem repository
2. Create a config file using the template (_config_files/config_local.ini_).
 Fill out the config file with all appropriate input and output paths, region information, and model parameters.
3. Select the appropriate run_XXX.py script to run based on your computing environment
4. Run the run_XXX.py script, passing the config file  
`python run_local.py -c config_files/my_config_file.ini -v None`
