"""Scans all the IPs from the Planet Lab EDNS set and 
saves the results in a Pickle:
    {ip: provider: [reverse_dns_hostnames]}"""

#Global
import dns
import dns.reversename
import dns.resolver
import socket

#Local
import edns

IPS_NA = [
    ("132.239.17.224", "UCSD"),
    ("129.110.125.52", "UTDallas"),
    ("128.223.8.114", "UOregon"),
    ("131.247.2.248", "USF"),
    ("198.108.101.61", "EMich"),
]

IPS_AS = [
    ("137.189.98.210", "CUHK"),
    ("222.197.180.139", "UESTC"),   
    ("202.112.28.98", "SJTU - Harbin"),
    ("143.248.55.129", "KAIST"),
    ("219.243.208.62", "Tsinghua University"),
]

IPS_EU = [
    ("188.44.50.106", "Moscow State University"),
    ("130.192.157.138", "DIT Italy"),
    ("62.108.171.76", "KUT Poland"),
    ("129.69.210.97", "UStuttgart"),
    ("130.237.50.124", "KTH RIT"),
]

IPS_SA = [
    ("200.19.159.34", "UFMG - BR"),
    ("200.17.202.194", "C3SL - BR"),
    ("190.227.163.141", "ITBA - AR"),
    ("157.92.44.101", "UBA - AR"),
    ("200.0.206.169", "Redclara - BR"),
]

IPS_OC = [
    ("130.194.252.8", "Monash University"),
    ("156.62.231.242", "AUT University - NZ"),
    ("130.195.4.68", "Victoria University Wellington"),
    ("139.80.206.132", "UOtago - NZ"),
    ("130.217.77.2", "Waikato"),
]

providers = {
    "google": "www.google.com",
    "edgecast": "gp1.wac.v2cdn.net",
    "alibaba": "img.alicdn.com",
    "cloudfront": "st.deviantart.net",
    "cdn77": "922977808.r.cdn77.net",
    "cdnetworks": "cdnw.cdnplanet.com.cdngc.net",
    "adnxs": "ib.adnxs.com"
}

GOOGLE = '8.8.8.8'

#edns.do_query(resolver, query, origin, mask, timeout=1.0)
def edns_all_ips():
    res_ips = {}
    for prov in providers.keys():
        res_ips[prov] = {}
        for origin in IPS_NA + IPS_AS + IPS_EU + IPS_SA + IPS_OC:
            result = edns.do_query(GOOGLE, providers[prov], origin[0], 32, timeout=4.0)
            res_ips[prov][origin[0]] = [x.split(' ')[2] for x in result['records']]
    rev_dns = {}
    for prov in providers.keys():
    	rev_dns[prov] = {}
        for origin in res_ips[prov].keys():
            for ip in res_ips[prov][origin]:
                try:
                    rev_lookup = dns.reversename.from_address(ip)
                    rev_result = str(dns.resolver.query(rev_lookup, "PTR")[0])
                except (dns.resolver.NXDOMAIN, dns.exception.SyntaxError, dns.resolver.NoNameservers) as e:
                    rev_result = e
                rev_dns[prov][ip] = rev_result
    return rev_dns

if __name__ == "__main__":
    print("dns scanner")
    all_rev_dns = edns_all_ips()
    for key in all_rev_dns['edgecast'].keys():
        print(key, all_rev_dns['edgecast'][key])
