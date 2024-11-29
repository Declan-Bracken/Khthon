input_file = 'requirements.txt'
output_file = 'requirements_clean.txt'

with open(input_file, 'r') as infile:
    lines = infile.readlines()

with open(output_file, 'w') as outfile:
    for line in lines:
        if 'file:///' not in line:
            outfile.write(line)

print(f"Cleaned requirements saved to {output_file}")
