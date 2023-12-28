import re
import json

# Read the content from the "countries.txt" file
with open("countries.txt", "r", encoding="utf-8") as txt_file:
    content = txt_file.read()

# Extract the country name and code using regular expressions
countries = re.findall(r'{name: \'(.*?)\', code: \'(.*?)\'}', content)

# Create a list of dictionaries in the correct JSON format
json_data = [{"name": name, "code": code} for name, code in countries]

# Write the JSON data to a new file called "countries.json"
with open("countries.json", "w", encoding="utf-8") as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=2)

print("Successfully converted and saved as 'countries.json'")
