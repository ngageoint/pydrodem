
# pydrodem: Advanced Hydrological Conditioning of DEMs

A brief description of what this project does and who it's for

### Introduction:
Welcome to ***'pydrodem'***, a cutting-edge toolkit specialized in the hydrological conditioning of digital elevation models (DEMs). Developed with Python, this toolkit encapsulates powerful methodologies that transform raw DEMs into hydrologically accurate representations, which are crucial for various geospatial analyses, including watershed delineation, flow accumulation, and flood simulation.

### Core Features:
* **DEM Conditioning:** At its heart, **'pydrodem'** focuses on hydrological conditioning, which corrects DEMs by filling sinks, removing spikes, and ensuring a continuous flow direction across the landscape.
* **Integration with Renowned Libraries:** By leveraging the capabilities of Geopandas, WhiteBoxTools, and TauDEM, **'pydrodem'** not only offers high-resolution DEM processing but also ensures accuracy and efficiency in its operations.
* **Parallel Computing Support:** Designed with scalability in mind, **'pydrodem** can harness the power of parallel computing environments. Whether you're running simulations on a multi-core server or a single desktop, this toolkit adjusts seamlessly.
* **Cloud Execution:** Transitioning to cloud platforms? No problem! **'pydrodem'** is equipped to operate within cloud environments, providing flexibility and scalability as your data grows.

### Development & Origin:
Born out of the need for transparent and reproducible hydrological analysis, **'pydrodem'** was initiated and developed at the National Geospatial-Intelligence Agency (NGA). As it was crafted by federal government employees, it enjoys a public domain status within the United States. This ensures that the toolkit is not only accessible to all but also invites collaborative enhancements.

### Technical Details:
**'pydrodem'** is architected to be modular and extensible. It encapsulates the hydrological conditioning processes into logical workflows, ensuring that each step, from data ingestion to post-processing, is coherent and traceable. This modularity also means that as newer conditioning methodologies or tools emerge, they can be integrated with minimal friction.

## Transparency

NGA is posting code created by government officers in their official duties in transparent platforms to increase the impact and reach of taxpayer-funded code. NGA is also posting pydrodem to give NGA partners insight into how elevation data is quality controlled within NGA to help partners understand parts of quality assurance checks.

### Pull Requests

If you'd like to contribute to this project, please make a pull request. We'll review the pull request and discuss the changes. This project is in the public domain within the United States and all changes to the core public domain portions will be released back into the public domain. By submitting a pull request, you are agreeing to comply with this waiver of copyright interest. Modifications to dependencies under copyright-based open source licenses are subject to the original license conditions.

### Dependencies:
Ensuring robust and efficient operations, **'pydrodem'** stands on the shoulders of giants, integrating several powerful Python libraries:

* **Geospatial Libraries:** Geopandas, GDAL, and Shapely handle the core geospatial data operations.
* **Hydrological Analysis:** WhiteBoxTools and TauDEM are specialized tools that bring advanced hydrological analysis capabilities.
* **Cloud Integration:** Boto3 and Botocore facilitate operations within AWS environments.
* **Data Handling & Analysis:** Numpy, Scipy, and Pandas provide foundational data manipulation capabilities, ensuring efficient matrix operations and data transformations.
* **Visualization:** Matplotlib assists in visualizing results and intermediate data, aiding in quality checks and presentations.

Getting Started:
To harness the power of 'pydrodem':

1 . Clone or download the pydrodem repository to your local machine. 
2 . Set up your environment by populating the provided configuration file (config_files/config_local.ini) with the appropriate paths, region specifics, and model parameters.
3 . Depending on your infrastructure, choose the corresponding run_XXX.py script.
4 . Launch the script, referencing your configuration:
```
python run_local.py -c config_files/my_config_file.ini -v None
```
