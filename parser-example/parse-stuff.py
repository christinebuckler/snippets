"""
simple example of you might write a parser
"""

import csv

## create the file handles
fidin1 = open("foo1.csv","r")
fidin2 = open("foo2.csv","r")
fidout = open("parsed.csv","w")

## use the csv module to read/write
reader1 = csv.reader(fidin1)
reader2 = csv.reader(fidin2)
writer = csv.writer(fidout)
header1 = reader1.__next__()
header2 = reader2.__next__()

## read in the first file that you need to reference
my_dict = {}
linecount = 0
for line in reader1:
    my_dict[linecount] = line
    linecount +=1 

## function definations of stuff that you need to do
def do_stuff(key,line):
    if key in my_dict:
        newline = [str("%s%s"%(i[0],i[1])) for i in zip(line,my_dict[key])]
    else:
        newline = ['na' for i in line]
    return newline

## go through the second file and combine the information        
linecount = 0
for line in reader2:
    newline = do_stuff(linecount,line)
    writer.writerow(newline)
    
## clean up
fidin1.close()
fidin2.close()
fidout.close()
print("done")
