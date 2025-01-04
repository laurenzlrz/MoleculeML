"""
Package contanins modules to load datasets (e.g. MD17).
For each dataset, there is a data loader and data wrapper class.

Data loader responsibilities:
- Load data from files
- Configure which data to load
- Should unify data origins

Data wrapper responsibilities:
- Store data in a structured way
- Provide methods to access data
- For each dataset the data wrapper should provide default methods to access the data (Dataframes, Arrays, GeoObject)
- Should unify the data access for different datasets
"""