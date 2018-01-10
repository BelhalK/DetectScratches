import urllib
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-o", "--browse", dest="browser",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-i", "--index", dest="index",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

(options, args) = parser.parse_args()

for element in args:
  with open(str(element), "r") as ins:
      array = []
      count=1
      for line in ins:
      	count+=1
        urllib.urlretrieve(str(line), "living_room/picture_"+str(count)+".jpg")


