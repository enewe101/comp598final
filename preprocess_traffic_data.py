from bs4 import BeautifulSoup
import os
import json

COUNTS_FILE = '../intersection_usage/vehicles_and_pedestrian_counts.xml'
JSON_FILE = '../intersection_usage/vehicles_and_pedestrian_counts.json'

PLACEMARKS_DIR = '../intersection_usage/placemarks/'

def explode_placemarks():
	'''
		The huge kml file is unweildy.  Extract each Placemark tag into 
		a separate file to make it easier to randomly access them.
	'''

	# read the traffic counts file into memory
	traffic_counts_fh = open(COUNTS_FILE, 'r')
	soup = BeautifulSoup(traffic_counts_fh.read(), 'xml')
	placemarks = soup.find_all('Placemark')
	job_size = float(len(placemarks))

	# iterate over all the placemarks, extract them to file
	i = 0
	for placemark in placemarks:

		# show progress
		if i % 100 == 0:
			print 100 * i / job_size, '%'

		# write the placemark to its own file
		write_fh = open(os.path.join(PLACEMARKS_DIR, '%d.xml' % i), 'w')
		write_fh.write(placemark.prettify().encode('utf8'))
		write_fh.close()

		# keep track of progress
		i += 1


def extract_usage_info():
	'''
		Once the kml file is broken into smaller files, iterate over them,
		extracting the relevant information from them.
	'''
	fnames = os.popen('ls %s' % PLACEMARKS_DIR).read().split()
	job_size = float(len(fnames))

	# start a json list of entries
	json_encoded_fh = open(JSON_FILE, 'w')
	json_encoded_fh.write('[\n')
	json_encoded_fh.close()

	# read each placemark file and convert it into an entry in the json file
	i = 0
	for fname in fnames:

		# show progress
		if i % 100 == 0:
			print '%2.1f %%' % (100 * i / job_size)

		# read and parse the placemark (intersection)
		path = os.path.join(PLACEMARKS_DIR, fname)
		parsed = parse_placemark(open(path, 'r'))

		# write an entry to the json file (one entry per line)
		json_encoded_fh = open(JSON_FILE, 'a')
		json_encoded_fh.write('%s,\n' % json.dumps(parsed))
		json_encoded_fh.close()

		# track progress
		i += 1

	# close the json list
	json_encoded_fh = open(JSON_FILE, 'w')
	json_encoded_fh.write(']\n')
	json_encoded_fh.close()
	



def safe_str(s):
	return s.encode(errors='ignore')

DATA_TYPES = {
	'Date': safe_str,
	'Heure': safe_str,
	'Rue_1': safe_str,
	'Rue_2': safe_str,
	'Intersecti': safe_str,
	'Y': safe_str,
	'X': safe_str,
}

def parse_placemark(placemark_fh):

	placemark = BeautifulSoup(placemark_fh, 'xml')

	# extract the intersection coordinates
	coords = placemark.find('coordinates')
	lat, lon, alt =  [float(x) for x in coords.text.strip().split(',')]

	# extract the cdata which contains all of the traffic flow info
	cdata = placemark.find('description')

	# the cdata is itself an embedded html file, parse it.
	cdata = BeautifulSoup(cdata.string)

	# find the second table, it has the goods.
	table = cdata.find_all('table')[1]

	# iterate over the rows.  They organize the info in a a key-value pattern
	rows = table.find_all('tr')
	intersection_record = {'lat':lat, 'lon':lon}
	for row in rows:

		td1, td2 = row.find_all('td')
		key, val = td1.text, td2.text

		# parse certain fields differently
		if key in DATA_TYPES:
			val = DATA_TYPES[key](val)

		# by default, assume field is an integer
		else:
			val = int(val)

		intersection_record[key] = val

	return intersection_record








