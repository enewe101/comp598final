from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
import os
import json

COUNTS_FILE = '../intersection_usage/vehicles_and_pedestrian_counts.xml'
JSON_FILE = '../intersection_usage/vehicles_and_pedestrian_counts.json'
FLOWS_FILE = '../intersection_usage/flows.json'
PLACEMARKS_DIR = '../intersection_usage/placemarks/'

MONTREAL_ANGLE = 50	# angle by which Montreal "North" deviates from North

def analyze_usage():
	'''
		Inspect the spatial and temporal coverage of the data
	'''
	data = json.loads(open(JSON_FILE).read())
	dates = []

	# count the number of entries we have per year
	years = Counter()
	for datum in data:
		day, month, year = datum['Date'].split('/')
		years[year] += 1

	print '\n'.join(['%s: %s' % y for y in years.items()])
	return

	# 2009 has the most entries, Let's look closer at 2009 data
	data_2009 = filter(lambda datum: get_year(datum) == 2009, data)

	# Each intersection in 2009 gets covered about 32 times (rare exceptions)
	intersections = Counter()
	for datum in data_2009:
		intersection = datum['Intersecti']
		intersections[intersection] += 1

	# there are 923 different intersections
	num_intersections = len(intersections)

	# Which intersections didn't get covered 32 times?  How many times?
	alt_intersections = dict(
		filter(lambda i: i[1] != 32, intersections.items())
	)

	# do all intersections have lat, lon defined?  Looks like yes.
	locations = {}
	for datum in data_2009:
		locations[datum['Intersecti']] = (datum['lat'], datum['lon'])

	# pick an intersection, and see when it gets covered
	# intersection = 'Parc / 30 m Sud de Duluth'
	intersection = intersections.keys()[24]
	datetimes = []
	datetimes = [
		(d['Date'], d['Heure']) 
		for d in data_2009 
		if d['Intersecti'] == intersection
	]
	# print '\n'.join([str(d) for d in datetimes])

	# For the same intersection, let's see the southbound counts
	counts = [
		(compute_outbound_flow(d), compute_inbound_flow(d))
		for d in data_2009 
		if d['Intersecti'] == intersection
	]

	# When does the strongest morning/evening traffic occur?
	# morning: entries 7,8,9
	# evening: entries 25,26,27
	counts = [
		[
			sum(compute_outbound_flow(d)) + sum(compute_inbound_flow(d))
			for d in data_2009 if d['Intersecti'] == intersection
		]
		for intersection in intersections
	]
	peeks = Counter([
		max(enumerate(c[-12:]), key=lambda q: q[1])[0]
		for c in counts
	])

	print '\n'.join([str(c) for c in peeks.items()])


def extract_vectors():
	'''
		There are (usually) 32 entries per intersection, which correspond
		to triffic measurements throughout the day.  For each intersection,
		extract traffic flows that summarize the AM and PM rushour flows.
		Also annotate this with the intersection name and location.
	'''

	# open a file in which to write results
	flows_fh = open(FLOWS_FILE, 'w')

	# get the 2009 data
	data = json.loads(open(JSON_FILE).read())
	data = filter(lambda d: get_year(d) == 2009, data)

	# make a list of unique intersections and their locations 
	intersections = dict([
		(
			d['Intersecti'], 		# key
			(d['lat'], d['lon'])	# val
		)
		for d in data
	])

	jobsize = float(len(intersections))

	# Augment this with the AM and PM traffic flow vectors for each 
	# intersection
	vectors = {}
	j = 0
	for i in intersections:

		# show progress
		if j % 100 == 0:
			print '%2.1f %%' % (100 * j / jobsize)
		j += 1

		lat = intersections[i][1]
		lon = intersections[i][0]
		v = extract_intersection_vector(i,data)
		try:

			am_net_flow = (
				v['AM_flow'][0] - v['AM_flow'][1],
				v['AM_flow'][2] - v['AM_flow'][3]
			)
			pm_net_flow = (
				v['PM_flow'][0] - v['PM_flow'][1],
				v['PM_flow'][2] - v['PM_flow'][3]
			)
			tide = (
				abs(am_net_flow[0] - pm_net_flow[0])
				+ abs(am_net_flow[1] - pm_net_flow[1])
			)
			bulk = (
				v['AM_flow'][0] + v['AM_flow'][1]
				+ v['AM_flow'][2] + v['AM_flow'][3]
				+ v['PM_flow'][0] + v['PM_flow'][1]
				+ v['PM_flow'][2] + v['PM_flow'][3]
			)

			# we're going to calculate 'tidality', it is a measure of the
			# proportionate amount the flow tends to reverse from am to pm
			am_norm = np.linalg.norm(am_net_flow)
			pm_norm = np.linalg.norm(pm_net_flow)
			renorm = ( 1 / max(am_norm, pm_norm) ) ** 2
			tidality = -1 * np.dot(am_net_flow, pm_net_flow) * renorm

			# the am_net_flow variables are the only ones retained which still
			# posess directionality (except for the raw counts passed through
			# as 'AM_flow' and 'PM_flow').  We need to rotate them to align
			# with "Montreal North"
			am_net_flow = rotate(*am_net_flow, theta_deg = MONTREAL_ANGLE)
			pm_net_flow = rotate(*pm_net_flow, theta_deg = MONTREAL_ANGLE)

			vectors[i] = {
				'latitude': lat,
				'longitude': lon,
				'tide': tide,
				'tidality': tidality,
				'bulk': bulk,
				'am_net_flow': am_net_flow,
				'pm_net_flow': pm_net_flow,
				'AM_flow': {
					'eastbound': v['AM_flow'][0],
					'westbound': v['AM_flow'][1],
					'northbound': v['AM_flow'][2],
					'southbound': v['AM_flow'][3],
				},
				'PM_flow': {
					'eastbound': v['PM_flow'][0],
					'westbound': v['PM_flow'][1],
					'northbound': v['PM_flow'][2],
					'southbound': v['PM_flow'][3],
				}
			}
		except IndexError:
			pass

	flows_fh.write(json.dumps(vectors, indent=2))
	flows_fh.close()

	return vectors


def extract_intersection_vector(i, data):

	# get all the flows for this intersection
	flows = [
		compute_inbound_flow(d) + compute_outbound_flow(d)
		for d in data if d['Intersecti'] == i
	]

	AM_flow = np.sum(flows[7:10], axis=0)
	PM_flow = np.sum(flows[25:28], axis=0)

	return {'AM_flow': AM_flow, 'PM_flow': PM_flow}


def compute_outbound_flow(entry):
	east_bound = entry['NG'] + entry['SD'] + entry['OTD']
	west_bound = entry['ND'] + entry['ETD'] + entry['SG']
	north_bound = entry['ED'] + entry['STD'] + entry['OG']
	south_bound = entry['NTD'] + entry['EG'] + entry['OD']

	return np.array([east_bound, west_bound, north_bound, south_bound])


def compute_inbound_flow(entry):
	east_bound = entry['OD'] + entry['OTD'] + entry['OG']
	west_bound = entry['ED'] + entry['ETD'] + entry['EG']
	north_bound = entry['SD'] + entry['STD'] + entry['SG']
	south_bound = entry['ND'] + entry['NTD'] + entry['NG']

	return np.array([east_bound, west_bound, north_bound, south_bound])



def get_year(d):
	return int(d['Date'].split('/')[2])

	

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
		extracting the relevant information from them.  Store this information
		in JSON format.
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
		if i>0:
			json_encoded_fh.write(',\n')
		json_encoded_fh.write(json.dumps(parsed))
		json_encoded_fh.close()

		# track progress
		i += 1

	# close the json list
	json_encoded_fh = open(JSON_FILE, 'w')
	json_encoded_fh.write('\n]\n')
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








def rotate(x, y, theta_deg):

	theta_rad = np.pi * theta_deg / 180.0
	cos = np.cos(theta_rad)
	sin = np.sin(theta_rad)

	new_x = x*cos - y*sin
	new_y = x*sin + y*cos

	return new_x, new_y
