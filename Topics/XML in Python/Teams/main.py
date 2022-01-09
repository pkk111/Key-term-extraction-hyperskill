#  write your code here 

from lxml import etree

tree = etree.parse('data/dataset/input.txt').getroot()

names = ""
for member in tree[0]:
    names += " " + member.get('name')
print(names)
