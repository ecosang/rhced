# rhced

**Residential heating and cooling energy disaggregation (rhced)**

This is accompanies code of paper [“Ham, S.-W., Karava, P., Bilionis, I., and Braun, J. (2021). Scalable and practical heating and cooling energy disaggregation from smart thermostat and meter data for eco-feedback design. Submitted to xx.”](https://dx.doi.org/). This is a simple demonstaration of heating and cooling (HC) energy disaggregation from the net energy consumption. 

Please see our demo notebook **[link](https://ecosang.github.io/rhced/sample_bldg.html)**. 

# Folder structure

- data: sample data
- docs: demo notebook in .html format [click](https://ecosang.github.io/rhced/sample_bldg.html).
- notebook: demo notebook based on the sample data.
- rhced: model code.
- outputs: model outputs.
- visualization: collection of R scripts to draw the figures presented in the paper.

# Model structure

```bash
├──rhced
│   ├──__init__.py
│   ├──data_utils.py: Utility functions for data processing. Used for sample data demo. 
│   ├──misc.py: some utility functions for data processing. No needs for sample data demo.
│   ├──prediction.py: prediction module.
│   ├──training.py: training module.
```

# required packages
The following Python packages are required.
- [`pymc3`>=3.11](https://github.com/pymc-devs/pymc3#installation)
- [`pyarrow`>=5.00](https://arrow.apache.org/docs/python/install.html)
- [`pandas`>=1.3.2](https://pandas.pydata.org/)
- [`numpy`>=1.21.2](https://numpy.org/)
- [`psypy`>=0.0.2](https://pypi.org/project/psypy/)

# Visualization

It is a collection of R scripts to draw the figures presented in the paper. 

You can open `rhced.Rproj` project, but the visualization functions are not part of the model.

The following R packages are required.

- [tidyverse](https://www.tidyverse.org/)
- [pathchwork](https://patchwork.data-imaginist.com/)
- [lubridate](https://lubridate.tidyverse.org/)

# Generate .html file

- `jupyter nbconvert --to html sample_bldg.ipynb`

Copy the generated .html to ~/docs/xx.html to make the latest notebook is published at https://ecosang.github.io/rhced/sample_bldg.html 

For exampe, in Window machine, `copy sample_bldg.html ../docs/sample_bldg.html`
