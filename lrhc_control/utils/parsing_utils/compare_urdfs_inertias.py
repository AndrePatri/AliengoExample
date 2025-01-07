import xml.etree.ElementTree as ET
import argparse

def parse_urdf(file_path):
    """
    Parses a URDF file and returns the robot name, link information, and inertial properties.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Get robot name
        robot_name = root.attrib.get('name')

        # Extract links and inertial properties
        links = {}
        for link in root.findall('link'):
            link_name = link.attrib.get('name')
            inertial = link.find('inertial')

            if inertial is not None:
                mass = float(inertial.find('mass').attrib.get('value', 0.0))
                inertia = inertial.find('inertia')
                inertia_matrix = {
                    'ixx': float(inertia.attrib.get('ixx', 0.0)),
                    'ixy': float(inertia.attrib.get('ixy', 0.0)),
                    'ixz': float(inertia.attrib.get('ixz', 0.0)),
                    'iyy': float(inertia.attrib.get('iyy', 0.0)),
                    'iyz': float(inertia.attrib.get('iyz', 0.0)),
                    'izz': float(inertia.attrib.get('izz', 0.0))
                }
            else:
                mass = 0.0
                inertia_matrix = None

            links[link_name] = {
                'mass': mass,
                'inertia': inertia_matrix
            }

        return robot_name, links

    except Exception as e:
        print(f"Error parsing URDF file '{file_path}': {e}")
        return None, None

def compare_links(links1, links2):
    """
    Compares the links and inertial properties of two URDF files.
    """
    missing_in_urdf2 = set(links1.keys()) - set(links2.keys())
    missing_in_urdf1 = set(links2.keys()) - set(links1.keys())

    # Print missing links
    if missing_in_urdf2:
        print("Links found in the first URDF but not in the second:")
        for link in missing_in_urdf2:
            mass = links1[link]['mass']
            inertia = links1[link]['inertia']
            inertia_str = f"Mass: {mass}, Inertia: {inertia}" if inertia else f"Mass: {mass}, Inertia: None"
            print(f"  - {link} ({inertia_str})")

    if missing_in_urdf1:
        print("Links found in the second URDF but not in the first:")
        for link in missing_in_urdf1:
            mass = links2[link]['mass']
            inertia = links2[link]['inertia']
            inertia_str = f"Mass: {mass}, Inertia: {inertia}" if inertia else f"Mass: {mass}, Inertia: None"
            print(f"  - {link} ({inertia_str})")

    # Compare inertial properties for common links
    common_links = set(links1.keys()) & set(links2.keys())
    for link in common_links:
        inertial1 = links1[link]
        inertial2 = links2[link]

        differences = []

        # Compare mass
        if inertial1['mass'] != inertial2['mass']:
            differences.append(f"Mass: {inertial1['mass']} vs {inertial2['mass']}")

        # Compare inertia matrix
        if inertial1['inertia'] != inertial2['inertia']:
            for key in inertial1['inertia']:
                if inertial1['inertia'][key] != inertial2['inertia'][key]:
                    differences.append(f"{key.upper()}: {inertial1['inertia'][key]} vs {inertial2['inertia'][key]}")

        if differences:
            print(f"Differences in link '{link}':")
            for diff in differences:
                print(f"  - {diff}")

def compare_urdfs(urdf1_path, urdf2_path):
    """
    Compares two URDF files and prints differences in inertial properties and missing links.
    """
    print("Parsing first URDF...")
    robot_name1, links1 = parse_urdf(urdf1_path)

    print("Parsing second URDF...")
    robot_name2, links2 = parse_urdf(urdf2_path)

    if robot_name1 is None or robot_name2 is None:
        print("Error: Failed to parse one or both URDF files.")
        return

    # Check robot names
    if robot_name1 != robot_name2:
        print(f"Warning: Robot names do not match ({robot_name1} vs {robot_name2})")
    else:
        print(f"Robot name: {robot_name1}")

    print("\nComparing links...")
    compare_links(links1, links2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two URDF files for inertial differences.")
    parser.add_argument('urdf1', type=str, help="Path to the first URDF file.")
    parser.add_argument('urdf2', type=str, help="Path to the second URDF file.")

    args = parser.parse_args()

    compare_urdfs(args.urdf1, args.urdf2)

