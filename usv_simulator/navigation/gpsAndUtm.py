import utm
import numpy as np

def get_utm(lon, lat):
    [east, north, zone_number, zone_letter] = utm.from_latlon(latitude=lat, longitude=lon)
    # print('east is %0.3f m,north is %0.3f m' % (east, north))
    return [east, north, zone_number, zone_letter]


def get_wgs84(east, north, zone_number, zone_letter):
    lat, lon = utm.to_latlon(east, north, zone_number, zone_letter)
    # print('lon is %0.8f, lat is %0.8f' % (lon, lat))
    return [lon, lat]


def get_list_wgs84(init_gps84, points_xyz):
    points_wgs84 = np.zeros(shape=[len(points_xyz), 2])
    local = get_utm(init_gps84[0], init_gps84[1])
    for var in range(len(points_xyz)):
        points_wgs84[var, :] = get_wgs84(east=(local[0] + points_xyz[var][1]),
                                         north=(local[1] + points_xyz[var][0]),
                                         zone_number=local[2],
                                         zone_letter=local[3])

    return points_wgs84
