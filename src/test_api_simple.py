from sec_edgar_downloader import Downloader

# Test to find the correct method signature
downloader = Downloader("Northeastern University", "test@northeastern.edu", "data/raw")

# Check what methods are available
print("Available methods:")
methods = [method for method in dir(downloader) if not method.startswith('_')]
print(methods)

# Get help on the get method
print("\nHelp for get method:")
help(downloader.get)