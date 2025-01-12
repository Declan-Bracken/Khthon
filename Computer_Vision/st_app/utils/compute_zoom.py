import math

def zoom_from_sqkm(sqkm):
    return math.log2(40000 / (sqkm / 2))

def sqkm_from_zoom(zoom):
    return (40000 / (2 ** zoom)) * 2

if __name__ == "__main__":
    # Test code
    zoom = 10
    sqkm = sqkm_from_zoom(zoom)
    print(f'sqkm = {sqkm}')
    print(f'zoom = {zoom_from_sqkm(sqkm)}')
