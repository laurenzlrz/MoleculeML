"""
General Data Pipeline Structure:

1. Loader: Responsible for loading data from various data paths, setting appropriate units and attributes,
and storing the data in a data object.

2. AbstractDataObject: A structured data storage class that can be flattened and converted
to pandas DataFrame or numpy arrays. It ensures two types of attributes:
generic attributes for the entire object and instance-specific attributes.
This class allows easy access to attributes and units and facilitates performing calculations.

3. Concrete DataObject: A more structured data storage class tailored for specific domains.
For example, it can store atom numbers separately to perform domain-specific calculations.

4. Functional DataObjects: These objects store data in forms specific to particular use cases.
For instance, a geometry object can be used to perform geometric calculations.

The specificity of the data storage and handling increases progressively from 1 to 4.
"""