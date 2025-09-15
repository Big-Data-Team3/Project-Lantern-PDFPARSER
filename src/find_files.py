from pathlib import Path

# Check different possible locations
locations_to_check = [
    Path("data/raw"),
    Path("../data/raw"),
    Path("."),
    Path("AAPL"),
    Path("data")
]

print("Searching for downloaded files...")

for location in locations_to_check:
    if location.exists():
        files = list(location.rglob("*.htm"))
        if files:
            print(f"\nFound files in {location.absolute()}:")
            for file in files:
                size = file.stat().st_size / (1024*1024)
                print(f"  {file.name} ({size:.1f} MB)")