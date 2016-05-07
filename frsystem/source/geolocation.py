"""Functions for implementing geolocation and physical distance estimations"""

#Global
import math

#Local

def get_phys_distance(c1, c2):
    """Takes too cordinate lists in format [lat, long] and estimates physical distance in kilometers
    Adapted from pwtools.py in oak-tools repository"""
    lat1 = float(c1[0])
    lat2 = float(c2[0])
    lon1 = float(c1[1])
    lon2 = float(c2[1])
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = lon1*degrees_to_radians
    theta2 = lon2*degrees_to_radians
    cos= (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) + math.cos(phi1)*math.cos(phi2))
    arc = math.acos(cos)
    return arc*6373
