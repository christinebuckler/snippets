##
# Usage:
# python wordle_csv_upload.py /tmp/file.csv
##

import webbrowser
import sys, os

filename = sys.argv[1]
csv_file = open(filename)

result_file_name = "/tmp/hede.html"
result_file = open(result_file_name, "w")


html_template = """
<form action="http://www.wordle.net/advanced" method="POST">
    <textarea name="text">%text%</textarea>
    <input type="submit">
</form>
"""

text = ""

for line in csv_file:
    line = line.replace("\n", "")
    for part in line.split(","):
        text += " " + part

# print(text)
html_template = html_template.replace("%text%", text)
result_file.write(html_template)
result_file.close()
csv_file.close()

# webbrowser.open(result_file_name)
webbrowser.open('file://' + os.path.realpath(result_file_name))
