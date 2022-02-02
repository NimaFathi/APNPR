import glob

# Contains all .txt files except our list of classes
txt_files = [file for file in glob.glob('IRCP_dataset_1024X768/*.txt') if file != 'images/classes.txt']

# Read every .txt file and store it's content into variable curr
for file in txt_files:
    print(file)
    with open(file, 'r') as f:
        curr = f.read()

# Replace class index 15 with 1 and store it in a variable new
new = curr.replace('0 ', '1 ')

# Once again open every .txt file and make the replacement
for file in txt_files:
    with open(file, 'w') as f:
        f.write(new)
