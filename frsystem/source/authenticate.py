"""Scripts for interacting with Planet Lab's authentication system"""

#Global
import sys
import xmlrpclib

#Local

def authenticate(server, username, password):
    """This function takes the username, password, and the api server URL and returns the xmlrpclib server and auth objects
    Based on example provided at https://www.planet-lab.org/doc/plcapitut#id2"""

    api_server = xmlrpclib.ServerProxy(server, allow_none=True)

    auth = {}
    auth['AuthMethod'] = 'password'
    auth['Username'] = username
    auth['AuthString'] = password

    if (api_server.AuthCheck(auth)):
        return api_server, auth
    else:
        print("Planet-Lab API authorization failed.")
        sys.exit(-1)

if __name__ == "__main__":
    server = "https://www.planet-lab.org/PLCAPI/"
    username = "marcelflores2007@u.northwestern.edu"
    password = "EKNp+H3HmWYUsoH3zoPexvXVF5ezD7s+NBZ10gutv0I="
    authenticate(server, username, password)
    print("Successfully connected to PL API server")
