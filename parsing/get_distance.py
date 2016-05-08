import math
#Approx radius of Earth in meters
Rad = 6371000

def get_dist(lat1, long1, lat2, long2):
    """Finds the great circle distance between a pair of coordinates using the haversine formula
       :param lat1: Latitude of first point represented as a decimal number in degrees (negative denotes South)
       :param long1: Longitude of first point represented as a deciman number in degrees (negative denotes West)
       :param lat2: Latitude of second point (represented same as lat1)
       :param long2: Longitude of second point (represented same as long1)

       :return: Distance between point (lat1, long1) and point (lat2, long2) in meters
    """
    
    # Variable setup
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    delta_p = math.radians(lat2-lat1)
    delta_l = math.radians(long2-long1)

    # Setup first and last terms
    term_1 = math.sin(delta_p/2.0)
    sq_term_1 = term_1*term_1

    term_2 = math.sin(delta_l/2.0)
    sq_term_2 = term_2*term_2

    # Calculate c of haversine formula
    a = sq_term_1 + math.cos(p1)*math.cos(p2)*sq_term_2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))

    # distance is Radius of earth * calcualted c
    return Rad*c
   
if __name__ == "__main__":
    # distance between lax, ord
    # d = get_dist(41.9742, -87.9073, 33.9416, -118.4085)
    # distance between nyc, ord
    d = get_dist(41.9742, -87.9073, 40.7128, -74.0059)
    print(d)
