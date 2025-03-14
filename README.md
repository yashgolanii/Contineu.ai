# RPLidar [![Documentation](https://readthedocs.org/projects/rplidar/badge/?version=latest)](http://rplidar.readthedocs.org/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/rplidar.svg)](https://pypi.python.org/pypi/rplidar) [![MIT License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/SkRobo/rplidar/blob/master/LICENSE)




## Installing

You can install rplidar using `pip`:

```sh
$ pip install rplidar-roboticia
```

Or for Python 3:
```sh
$ sudo pip3 install rplidar-roboticia
```

## Documentation

View the latest rplidar documentation at http://rplidar.rtfd.org/.

## Usage example

Simple example:
```Python
from rplidar import RPLidar
lidar = RPLidar('/dev/ttyUSB0')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)

for i, scan in enumerate(lidar.iter_scans()):
    print('%d: Got %d measurments' % (i, len(scan)))
    if i > 10:
        break

lidar.stop()
lidar.stop_motor()
lidar.disconnect()
```

In addition to it you can view example applications inside
[examples](https://github.com/SkRobo/rplidar/tree/master/examples>) directory.
