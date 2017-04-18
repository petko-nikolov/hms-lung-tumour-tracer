import argparse
import os
import glob
import re
from collections import defaultdict
import json
from utils import mm_to_pixels, STRUCTURE_CLASS
import csv

SEED_OFFSET = 20


def parse_tags(tags):
    space_per_pixel = [float(x) for x in tags['0028.0030'].split(',')]
    tags['dx'] = space_per_pixel[0]
    tags['dy'] = space_per_pixel[1]
    slice_start = [float(x) for x in tags['0020.0032'].split(',')]
    tags['x0'] = slice_start[0]
    tags['y0'] = slice_start[1]
    return tags


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert to dataset.')
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Input directory.')
    parser.add_argument(
        '--output_file', type=str, required=True,
        help='output_file')
    parser.add_argument(
        '--seed_file', type=str, required=True,
        help='Seed file.')
    args = parser.parse_args()

    seeds = defaultdict(lambda: defaultdict(list))

    # read seed file
    with open(args.seed_file, 'r') as rf:
        for row in csv.reader(rf, delimiter=','):
            seeds[row[0]][row[1]].append((float(row[2]), float(row[3])))

    with open(args.output_file, 'w') as wf:
        for scan_path in glob.glob(args.input_dir + '/*'):
            scan = os.path.basename(scan_path)
            structure_to_index = defaultdict(list)
            if os.path.exists(os.path.join(scan_path, 'structures.dat')):
                with open(os.path.join(scan_path, 'structures.dat'), 'r') as f:
                    structures = [s.strip() for s in f.read().split('|')]
                    for i, st in enumerate(structures):
                        matching = [mst for (mst, _) in STRUCTURE_CLASS if mst in st.lower()]
                        if matching:
                            structure_to_index[matching[0]].append(i)

            print(structure_to_index)
            for slice_image_path in glob.glob(scan_path + '/pngs/*.png'):

                slice_id = re.match('(\w+)\.png',
                                    os.path.basename(slice_image_path)).groups()[0]

                with open(os.path.join(scan_path, 'auxiliary', slice_id + '.dat')) as f:
                    slice_data = {}
                    for pair in f:
                        key, value = re.match("\((\d+\.\d+)\),(.*)", pair).groups()
                        slice_data[key] = value
                    tags = parse_tags(slice_data)

                slice_seeds = []
                for seed_slice in seeds.get(scan, {}).keys():
                    if (int(seed_slice) - SEED_OFFSET <= int(slice_id) <= int(seed_slice) +
                            SEED_OFFSET):
                        for point in seeds[scan][seed_slice]:
                            x, y = mm_to_pixels(point[0],
                                                point[1],
                                                tags['x0'],
                                                tags['y0'],
                                                tags['dx'],
                                                tags['dy'])
                            slice_seeds.append((round(x), round(y)))

                if not slice_seeds:
                    continue

                result = defaultdict(list)
                for structure, indexes in structure_to_index.items():
                    if structure != 'radiomics_gtv':
                        continue
                    for index in indexes:
                        polygon_path = os.path.join(
                            scan_path, 'contours',
                            slice_id + '.' + str(index + 1) + '.dat')

                        if os.path.exists(polygon_path):
                            with open(polygon_path) as f:
                                for polygon_line in f:
                                    pixel_coordinates = []
                                    physical_coordinates = [
                                        float(x) for x in polygon_line.split(',')]
                                    for i in range(0, len(physical_coordinates), 3):
                                        x, y = mm_to_pixels(physical_coordinates[i],
                                                            physical_coordinates[i + 1],
                                                            tags['x0'],
                                                            tags['y0'],
                                                            tags['dx'],
                                                            tags['dy'])
                                        pixel_coordinates.extend([round(x), round(y)])
                                    result[structure].append(pixel_coordinates)
                final_result = {'scan_id': scan, 'slice_id': slice_id,
                                'structures': result, 'tags': tags, 'seeds': slice_seeds}
                wf.write("{}\n".format(json.dumps(final_result)))
