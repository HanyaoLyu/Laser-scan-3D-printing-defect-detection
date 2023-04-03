"""GCODE Parser

Parses a GCODE file and creates a model reconstruction split into layers and lines.
Currently only supports GCODE with linear moves (G0/G1) in absolute mode.

The parse_gcode() function takes a GCODE file and turns it into a Model object.
"""

import re
import numpy as np
import time
import math
from collections import Counter

from metpy.calc import reduce_point_density
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN #Clustering


#TODO: Remove all references to matplot lib (its just used for development/debugging not the actual plugin)
try:
	import matplotlib.pyplot as plt

except:
	print("Matplotlib not installed")

class Model():
    """Model class for storing layer and max/min data"""
    
    def __init__(self, layers, max_x, max_y, max_z, min_x, min_y, min_z, z_hop_height):
        """
        Parameters
        ----------
        layers : [Layer]
            List of Layer objects
        max_x : float
            Maximum x value of model
        max_y : float
            Maximum y value of model
        max_z : float
            Maximum z value of model
        min_x : float
            Minimum x value of model
        min_y : float
            Minimum y value of model
        min_z : float
            Minimum z value of model
		z_hop_height : float
			Value of z increase when a z-hop is done
        """
        
        self.layers = layers
        self.max_y = max_y
        self.max_x = max_x
        self.max_z = max_z
        self.min_y = min_y
        self.min_x = min_x
        self.min_z = min_z
        self.layer_heights = [layer.z_height for layer in layers]
        self.z_hop_height = z_hop_height
        
    def to_svgs(self, dir_name):
        """Saves all layers as SVG images in directory. Used for testing only

        Parameters
        ----------
        dir_name : str
            Directory name to save images into 
        """
        
        for layer_index, layer in enumerate(self.layers):
            layer.to_svg(self.max_y, self.max_x, dir_name + "/{}.svg".format(layer_index))

class Line():
    """Line class defines a continuous extrusion in a layer"""
    
    def __init__(self):
        self.x = []
        self.y = []
        self.line_type = "regular"
        
    def append_coords(self, x, y):
        """Append a coordinate to an existing line

        Parameters
        ----------
        x : float
            x coordinate of point
        y : float
            y coordinate of point
        """

        # If this is first point in the line, append the coordinates to the line
        if len(self.x) == 0:
            self.x.append(x)
            self.y.append(y)
        else:
            prev_x = self.x[-1]
            prev_y = self.y[-1]
            # Compute the length between the new point and the last point
            segment_length = np.sqrt((x-prev_x)**2+(y-prev_y)**2)
            # Calculate number of points required to interpolate to a resolution of 0.1mm
            num_points = int(np.floor(segment_length/0.1))

            # Filter by segment length. Generally, infills are long lines defined by 2 points, whereas
            # shells are defined by regularly spaced points. 
            if segment_length > 2.5:
                if len(self.x) > 5:
                    self.line_type = "regular"
                else:
                    self.line_type = "infill"

            # Interpolation        
            if num_points < 2:
                # If the number of points to interpolate is less than 2, just append the points
                self.x.append(x)
                self.y.append(y)
            else:
                if x < prev_x:
                    xs = np.linspace(x, prev_x, num_points)
                    ys = np.interp(xs, [x, prev_x], [y, prev_y])
                    xs = np.flip(xs)
                    ys = np.flip(ys)
                else:
                    xs = np.linspace(prev_x, x, num_points)
                    ys = np.interp(xs, [prev_x, x], [prev_y, y])
                self.x.extend(xs[1:])
                self.y.extend(ys[1:])
            
    def len(self):
        """Returns number of points in line - 1"""
        return len(self.x) - 1
    
    def is_empty(self):
        """Checks if line is empty"""
        if self.len() > 0:
            return False
        else:
            return True
        
class Layer():
    """Layer class contains all lines in a layer, along with z-height info and
       functions for updating a layer and converting a layer to an SVG image."""
    
    def __init__(self, z_height):
        """
        Parameters
        ----------
        z_height : float
            The height of the layer
        """
        
        self.lines  = []
        self.new_line()
        self.prev_e = 0
        self.prev_g = -1
        self.probe_reach = 3.5 #Vertical difference in probe height when activated vs. not activated.
        self.z_height = z_height
        self.sample_points = dict([('x', []), ('y', []), ('material', [])]) #Stores x,y coordinates and a true/false value if material should be detected or not
        self.shift = 1.5 #Distance away from boundary to sample when testing for layer shifting
        self.has_z_hop = False
        self.points_sampled = False
        
    def append_coords(self, x, y, e, g):
        """
        Appends coordinates to a line as appropriate. Checks if a command is an extrusion,
        if it is, adds the coordinates to a line. Also checks if the line is continuous,
        if it is not, starts a new line.
        
        Parameters
        ----------
        x : float
            X-coordinate of command
        y : float
            Y-coordinate of command
        e : float
            Extursion value
        g : int
            G-CODE command type (not used)
        """
        #TODO: BUG FIX - This check does not account for possible filament retraction!
        if e > self.prev_e:                                 # If extrusion is taking place
            self.lines[-1].append_coords(x,y)               # Append coordinate to line
            self.prev_e = e
            self.prev_g = g
        else:                                               # If no extrusion is taking place
            if not self.lines[-1].is_empty():               # And the line object is not empty
                self.new_line()                             # Start a new line
            self.lines[-1].x = [x]
            self.lines[-1].y = [y]
                
    def new_line(self):
        """Makes a new line"""
        self.lines.append(Line())

	#This function is not used in this plugin
    def to_svg(self, max_height, max_width, fn):
        """
        Saves a layer as an SVG image file

        Parameters
        ----------
        max_height : float
            Maximum height of model in mm
        max_width : float
            Maximum width of model in mm
        fn : str
            Filename to save image to
        """
        
        with open(fn, "w") as f:
            f.write(('<svg id="svgImage" xmlns="http://www.w3.org/2000/svg"'
                     ' xmlns:xlink="http://www.w3.org/1999/xlink"'
                     ' viewBox="0 0 250 40" height="{}mm" width="{}mm">\n').format(max_height, max_width))
            for line in self.lines[::10]:
                points = '\t<polyline points="'
                coords = [f"{x},{y} " for x,y in zip(line.x, line.y)]
                points = points + "".join(coords) + ('"\n\tstyle="fill:none;stroke:black;stroke-width:0.4;'
                                                     'stroke-linejoin:round;stroke-linecap:round" />\n')
                f.write(points)
            for point in zip(self.sample_points['x'], self.sample_points['y']):
                f.write('\t<circle cx="{}" cy="{}" r="1" stroke="red" stroke-width="0" fill="red"></circle>\n'.format(point[0], point[1]))
            f.write('</svg>')
            
    def to_svg_inline(self, max_height, viewbox_width, viewbox_height):
        """
        Returns a layer as an SVG image

        Parameters
        ----------
        max_height : float
            Maximum height of model in mm
        viewbox_width : float
            Width of the SVG viewbox
        viewbox_height : float
            Height of the SVG viewbox
        """

        # Append the SVG header first
        out = ('<svg id="svgImage" xmlns="http://www.w3.org/2000/svg"'
                ' xmlns:xlink="http://www.w3.org/1999/xlink"'
                ' viewBox="0 0 {} {}" height="{}" width="100%">\n').format(viewbox_width, viewbox_height, max_height)
        #Transform all points and labels
        out += '<g transform=translate(1,187)>\n'
		#Add axis labels (do not scale these)
        out += ('<g font-size="6" font-family="Verdana" >\n'
        '<text x="22" y="0.2">+x</text>/n'
        '<text x="0" y="-22">+y</text>\n'
        '</g>\n')
		#Scale the rest of the image (flip along x axis)
        out += '<g transform=scale(1,-1)>\n'
		#Add the axes
        out += ('<g fill="none" stroke="black" stroke-width="0.4">\n'
        '<line x1="0" y1="0.2" x2="20" y2="0.2" />\n'		#y is offset slightly from 0 so full thickness of x axis is shown
        '<line x1="0" y1="0.2" x2="0" y2="20" />\n'
        '</g>\n')
        # Add the lines. Iterate through each line
        for line in self.lines:
            points = '\t<polyline points="'
            coords = [f"{x},{y} " for x,y in zip(line.x, line.y)]
            if line.line_type == "regular":
                # Append the points with black color if regular extrusion
                points = points + "".join(coords) + ('"\n\tstyle="fill:none;stroke:black;stroke-width:0.4;'
                                                         'stroke-linejoin:round;stroke-linecap:round" />\n')
            if line.line_type == "infill":
                # Append the points with blue color if infill
                points = points + "".join(coords) + ('"\n\tstyle="fill:none;stroke:blue;stroke-width:0.4;'
                                                         'stroke-linejoin:round;stroke-linecap:round" />\n')
            out += points

        # Add circles for each sample point
        for point in zip(self.sample_points['x'], self.sample_points['y'], self.sample_points['material']):
            if point[2]:
                out += '\t<circle cx="{}" cy="{}" r="1" stroke="red" stroke-width="0" fill="red"></circle>\n'.format(point[0], point[1])
            else:
                out += '\t<circle cx="{}" cy="{}" r="1" stroke="green" stroke-width="0" fill="LimeGreen"></circle>\n'.format(point[0], point[1])
        out += '</g>\n'
        out += '</g>\n'
        out += '</svg>'
        return out
    
    def plot_layer(self, col_reg, col_infill):
        """ Plot a layer in Matplotlib. Used for testing

        Parameters
        ----------
        col_reg : str
                Colorspec for regular extrusion
        col_infill : str
                Colorspec for infill

        """
        x = 0
        for line in self.lines:
            if line.line_type == "regular":
                plt.plot(line.x, line.y, col_reg) #self.z_height,
            elif line.line_type == "infill":
                plt.plot(line.x, line.y, col_infill)
        # plt.show()

    def to_csv(self, fn):
        """ Saves points as as csv

        Parameters
        ----------
        fn : str
                Filename to save to
        """
        with open(fn, "w") as f:
            for line in self.lines:
                [f.write("{},{}\n".format(x, y)) for (x,y) in zip(line.x, line.y)]
             
    def len(self):
        """ Returns the number of lines in the layer """
        return len([line for line in self.lines if line.len() > 0])

    def get_points(self, ignore_infill = False, filter = False):
        """ Returns a dictionary of all points in the layer

        Keyword Arguments
        -----------------
        ignore_infill : bool (Optional)
                Set to True to ignore infills. Defaults as false
        """
        x_pts = []
        y_pts = []
        

        [x_pts.extend(line.x) for line in self.lines if ignore_infill != True or line.line_type == "regular"]
        [y_pts.extend(line.y) for line in self.lines if ignore_infill != True or line.line_type == "regular"]

        x_ptsE = []
        y_ptsE = []
        

        [x_ptsE.extend(line.x) for line in self.lines if  line.line_type == "regular"]
        [y_ptsE.extend(line.y) for line in self.lines if  line.line_type == "regular"]
		#Filter points by reducing point density if specified
        if(filter):
            array = np.array([x_pts, y_pts])
            max_vals = np.argmax(array, axis=1) #Find max indices in x and y directions
            min_vals = np.argmin(array, axis=1) #Find min indices in x and y directions
            saved_indices = np.unique(np.concatenate((max_vals, min_vals))).tolist()
            priority = np.zeros(np.size(array, 1)) #Priority array for kept points during filtering
            for index in saved_indices:
                priority[index] = 1.0
                priority[index] = 1.0
            array = np.transpose(array)
            fil = reduce_point_density(array, 0.2, priority=priority) #Filter points with a 0.25 radius
            array = np.transpose(array[fil])
            x_pts = array[0, :]
            y_pts = array[1, :]

        return dict([('x', x_ptsE), ('y', y_ptsE), ('num_pts', len(x_ptsE))])
    def get_calibration_center(self):
        points = self.get_points()
        point_array = np.array([points['x'], points['y']])
        max_vals = np.max(point_array, axis=1)
        min_vals = np.min(point_array, axis=1)
        return max_vals.tolist(), min_vals.tolist(), ((max_vals + min_vals) / 2).tolist()



    def get_area_hull(self, xy):
        """ Returns the area of points xy using convex hull

        Arguments
        -----------------
        xy : 2d array
                Point cluster to find area using convex hull
        """
        point_list = []
        [point_list.append([x, y]) for x, y in zip(xy[:,0], xy[:,1])]
        features = np.array(point_list)

        # If there are less than 4 points, do nothing (These are likely the outlier points)
        if features.shape[0] > 4:
            try:
                hull = ConvexHull(features)  # Compute the convex hull
#                plt.plot(xy[:,0], xy[:,1], 'o')
#                for simplex in hull.simplices:
#                    plt.plot(features[simplex, 0], features[simplex, 1], '-k')
#                plt.show()
                return hull.volume #When working in 2D, "volume" is the area
            except:
                return 0.0 #Return 0 if an error occurs
        return 0.0

    def get_area_hull_cluster(self, points):
        """ Returns the area of all points specified after clustering using DBSCAN

        Arguments
        -----------------
        points : dict
                Points in the layer
        """
        area = 0
        points = np.transpose(np.array([points['x'], points['y']]))

# Compute DBSCAN
        db = DBSCAN(eps=3.0, min_samples=10).fit(points) #Cluster points with each point being within 2mm of another to be in same group
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
#        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#        n_noise_ = list(labels).count(-1)


# Plot results and find areas of each point cluster

# Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
        # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = points[class_member_mask & core_samples_mask]
            area += self.get_area_hull(xy)

        return area
    def gen_sample_points(self, sample_methods, layers, curr_layer, z_offset,printing_part):
        """ Generate sample points for this layer

        Sampling point algorithms live here. To add a new method, add a new elif section with an appropriate
        name. Can also add keyword arguments as needed.

        Parameters
        ----------
        sample_methods : dict
                Sampling methods to use (including layer shift sampling)
        layers : arr(Layers)
			Array of layers in the model
		curr_layer : int
			Current layer position to work with
        """

        # Get the points in this layer
        points = self.get_points(filter = True)
        x_pts = []
        y_pts = []
        material_det = [] #Stores true/false values whether material should be detected at that point
        area = self.get_area_hull_cluster(points) / 100.0 #Don't ignore infill points for more accurate results
        print(area)
 
        # if sample_methods=="MM":
        #     point_array = np.array([points['x'], points['y']])

        #     max_vals = np.argmax(point_array, axis=1)

        #     min_vals = np.argmin(point_array, axis=1)
        #     sample_pts = np.unique(np.concatenate((max_vals, min_vals))).tolist()
        #     x_pts.extend([points['x'][i] for i in sample_pts])
        #     y_pts.extend([points['y'][i] for i in sample_pts])
        #     plt.scatter(x_pts,y_pts)

        #     # print(list(zip(X,Y)))
        #     for a,b in zip(x_pts,y_pts): 
        #         a=round(a,2)
        #         b=round(b,2)
        #         plt.text(a, b, (str(a),str(b)),fontsize=10)
        #         print((a,b))
        #     plt.show()
        #     Sampoint1=(x_pts,y_pts)
        # return Sampoint1

        sample_pts = list(np.random.randint(0, points['num_pts'], int(area/0.2)))

        x_pts.extend([points['x'][i] for i in sample_pts])
        y_pts.extend([points['y'][i] for i in sample_pts]) 

        if sample_methods=="Min-max":
            # Put the points into a numpy array to make life a bit easier
            point_array = np.array([points['x'], points['y']])
            x=point_array[0]
            y=point_array[1]
            i=0
            # Find the minimum and maximum values along the regular axes
            max_vals = np.argmax(point_array, axis=1)

            min_vals = np.argmin(point_array, axis=1)

            """
            # Extra rotation which is not really necessary
            rot45 = np.array([[0.7071, -0.7071],[0.7071, 0.7071]])      # Transformation matrix for a rotation by 45 degrees
            point_array_rot = np.matmul(rot45, point_array)             # Rotate all points by 45 degrees
            max_vals_rot = np.argmax(point_array_rot, axis=1)           # Repeat process of finding min/max points
            min_vals_rot = np.argmin(point_array_rot, axis=1)
            """

            # Remove duplicate values if any
            sample_pts = np.unique(np.concatenate((max_vals, min_vals))).tolist()
            x_pts.extend([points['x'][i] for i in sample_pts])
            y_pts.extend([points['y'][i] for i in sample_pts])
            X=[x_pts[0]]
            Y=[y_pts[0]]

            i=0
            N=0
            print((max(x_pts)-min(x_pts))/3)  
            for i in range(0,len(x_pts)-1):
                for j in range(0,len(x_pts)-1):
                    D=abs(((x_pts[j]-x_pts[i])**2+(y_pts[j]-y_pts[i])**2)**(1/2))

                    if D<(max(x_pts)-min(x_pts))/(20*printing_part):
                        N=N+1
                        print([D,x_pts[i],y_pts[i]])
                if N<2:
                    X.append(x_pts[i])
                    Y.append(y_pts[i])
                N=0



            material_det.extend([True] * len(sample_pts))
            plt.scatter(X,Y)

            # print(list(zip(X,Y)))
            for a,b in zip(X, Y): 
                a=round(a,2)
                b=round(b,2)
                plt.text(a, b, (str(a),str(b)),fontsize=10)
                print((a,b))
            plt.show()
            Sampoint=(X,Y)
        return Sampoint


def parse_gcode(filename, logger, manager, identifier):
    """ Function for parsing GCODE file

    Parameters
    ----------
    filename : str
            File to process
    """
    layers  = []
    num_layers = 0
    layer_heights = []
    lines  = []
    Gs     = []
    xs     = []
    ys     = []
    zs     = []
    es     = []
    prev_x = 0
    prev_y = 0
    prev_z = 0
    prev_e = 0

    G92_warning_sent = False

    with open(filename) as f:                                           # Open GCODE file
        file = f.readlines()

    extrusion_mode = 1.0												# Modes are denoted as 1.0 for absolute positioning and 0.0 for relative
    position_mode = 1.0

    e_offset = 0.0														# Used to account for changes in origin from G92 command
    x_offset = 0.0
    y_offset = 0.0
    z_offset = 0.0

    z_hop_counter = 0													# Used to prevent adding unnecessary layers due to z hop (lifting z when switching layers). Set this to 1 to account for current command
    z_hop_bool = False													# Should counter be increasing? - T/F
    z_hop_values = []
    hop = 0
    z_hop_layers = set()
                    
    for line in file:                                                   # Iterate through lines
        line = line.strip()                                             # Strip newline character
        if not line or line[0] == ";":                                  # Skip empty lines and comments
            continue
        line = line.split(";")[0].strip()                               # Remove any trailing comments

        #TODO: Add handling for G2/G3

		#Set mode if command is G90, G91, M82, M83
        mode_match = re.match("^(?P<letter>[GM])(?P<number>\d{2})", line)
        if mode_match:
            if str(mode_match.group('letter')) == 'G' and int(mode_match.group('number')) == 90:
                extrusion_mode = 1.0
                position_mode = 1.0
            if str(mode_match.group('letter')) == 'G' and int(mode_match.group('number')) == 91:
                extrusion_mode = 0.0
                position_mode = 0.0
            if str(mode_match.group('letter')) == 'M' and int(mode_match.group('number')) == 82:
                extrusion_mode = 1.0
            if str(mode_match.group('letter')) == 'M' and int(mode_match.group('number')) == 83:
                extrusion_mode = 0.0
		#Set offset if origin is changed (G92 command)
        origin_match = re.match("^(?:G92)(\.\d+)?\s", line)
        if origin_match:
            x_change = re.match(".*X(-?\d*\.?\d*)", line)
            if x_change:
                x_offset += prev_x - float(x_change.group(1))
                prev_x = float(x_change.group(1))
                if not G92_warning_sent:								# G92 command detected! It will parse correctly but cannot be sampled. Send warning if one not already sent
                    message = dict(message="Warning: G92 command used with X, Y, or Z. Fault detection is not recommended as it may cause print to fail.")
                    manager.send_plugin_message(identifier, message)
                    G92_warning_sent = True
            y_change = re.match(".*Y(-?\d*\.?\d*)", line)
            if y_change:
                y_offset += prev_y - float(y_change.group(1))
                prev_y = float(y_change.group(1))
                if not G92_warning_sent:								# G92 command detected! It will parse correctly but cannot be sampled. Send warning if one not already sent
                    message = dict(message="Warning: G92 command used with X, Y, or Z. Fault detection is not recommended as it may cause print to fail.")
                    manager.send_plugin_message(identifier, message)
                    G92_warning_sent = True
            z_change = re.match(".*Z(-?\d*\.?\d*)", line)
            if z_change:
                z_offset += prev_z - float(z_change.group(1))
                prev_z = float(z_change.group(1))
                if not G92_warning_sent:								# G92 command detected! It will parse correctly but cannot be sampled. Send warning if one not already sent
                    message = dict(message="Warning: G92 command used with X, Y, or Z. Fault detection is not recommended as it may cause print to fail.")
                    manager.send_plugin_message(identifier, message)
                    G92_warning_sent = True
            e_change = re.match(".*E(-?\d*\.?\d*)", line)
            if e_change:
                e_offset += prev_e - float(e_change.group(1))
                prev_e = float(e_change.group(1))
        G_match = re.match("^(?:G0|G1)(\.\d+)?\s", line)                # Match for any linear movements
        if G_match:
            if z_hop_bool: z_hop_counter += 1							# Increment z_hop_counter
            layer_has_z_hop = False										# Boolean to control whether a Layer Object will have a z_hop after parsing completed
            x_loc = re.match(".*X(-?\d*\.?\d*)", line)					# Match all coordinate systems
            y_loc = re.match(".*Y(-?\d*\.?\d*)", line)
            z_loc = re.match(".*Z(-?\d*\.?\d*)", line)
            e_loc = re.match(".*E(-?\d*\.?\d*)", line)
            G = re.match("G([0123])", line)

            if z_loc and z_hop_bool and not e_loc:						# If layer changed without an extrusion on previous layer, remove commands from that layer
                es = remove_last_indices(es, z_hop_counter)
                xs = remove_last_indices(xs, z_hop_counter)
                ys = remove_last_indices(ys, z_hop_counter)
                temp_z = zs[-1]
                zs = remove_last_indices(zs, z_hop_counter)
                Gs = remove_last_indices(Gs, z_hop_counter)
                lines = remove_last_indices(lines, z_hop_counter)
                z_hop_counter = 0
                if len(zs) > 0 and temp_z - zs[-1] > 0:						# Make sure the z hop that is detected is an upwards motion
                    z_hop_layers.add(zs[-1])								# Add previous layer to z_hop_layers
                    z_hop_values.append(round(temp_z - zs[-1], 3))			# Add z hop height
                layer_has_z_hop = True

            if e_loc and z_hop_bool:									# If extrusion does take place, reset command counter and stop counting
                z_hop_bool = False
                z_hop_counter = 0

            if z_loc and not z_hop_bool and not e_loc:					# Layer changed, begin counting commands
                z_hop_bool = True

            lines.append(line)											# Append G and line
            Gs.append(int(G.group(1)))

            if e_loc and extrusion_mode > 0.5:							# Append E coordinates. If none, use prior
                e_loc = float(e_loc.group(1))
                prev_e = e_loc
                es.append(e_loc + e_offset)
            elif e_loc: #Relative ext
                e_loc = float(e_loc.group(1)) + prev_e
                prev_e = e_loc
                es.append(e_loc + e_offset)
            else:
                es.append(prev_e + e_offset)

            x_loc = re.match(".*X(-?\d*\.?\d*)", line)                  
            if x_loc and position_mode > 0.5:							# Append X coordinates. If none, use prior
                x_loc = float(x_loc.group(1))
                prev_x = x_loc
                xs.append(x_loc + x_offset)
            elif x_loc: #Relative pos
                x_loc = float(x_loc.group(1)) + prev_x
                prev_x = x_loc
                xs.append(x_loc + x_offset)
            else:
                xs.append(prev_x + x_offset)

            if y_loc and position_mode > 0.5:							# Append Y coordinates. If none, use prior
                y_loc = float(y_loc.group(1))
                prev_y = y_loc
                ys.append(y_loc + y_offset)
            elif y_loc: #Relative pos
                y_loc = float(y_loc.group(1)) + prev_y
                prev_y = y_loc
                ys.append(y_loc + y_offset)
            else:
                ys.append(prev_y + y_offset)

            if z_loc and position_mode > 0.5:							# Append Z coordinates. If none, use prior
                z_loc = float(z_loc.group(1))
                prev_z = z_loc
                zs.append(z_loc + z_offset)
            elif z_loc: #Relative pos
                z_loc = float(z_loc.group(1)) + prev_z
                prev_z = z_loc
                zs.append(z_loc + z_offset)
            else:
                zs.append(prev_z + z_offset)

    if len(z_hop_values) > 0:											# Find the maximum mode of a z_hop_values. This is the most likely value for a z_hop
        counted_hops = Counter(z_hop_values).most_common()				# Counts occurences of a value in a list of the form [ (value1, count1), (value2, count2) ...]
        hop = counted_hops[0][0]										# Initialize with first value in the list
        hop_count = counted_hops[0][1]
        for hop_height, count in counted_hops:							# If count is greater, it is the mode.
            if count > hop_count:
                hop = hop_height
                hop_count = count
            elif count == hop_count and hop_height > hop:				# If count is the same but values are different, take the higher value
                hop = hop_height
                hop_count = count
    [layer_heights.append(height) for height in set(zs)]                # Find the unique Z heights in the GCODE file
    layer_heights.sort()                                                # Make it a sorted list
    num_layers = len(layer_heights)                                     # Number of layers
    index = dict(zip(layer_heights, range(num_layers)))                 # Map layer height to an index

    layers = [Layer(layer_heights[i]) for i in range(num_layers)]       # Initialize model list with number of layers
    for i in range(len(lines)):                                         # Iterate through each line 
        layers[index[zs[i]]].append_coords(xs[i],ys[i],es[i],Gs[i])     # Append each command into the model list

    layers = [layer for layer in layers if layer.len() > 0]             # Find non-empty layers

    for layer in layers:												# If the first line in a layer is infill, remove the line from the layer (BUG FIX WORKAROUND - Likely due to retraction bug specified above)
        if layer.lines[0].line_type == "infill":
            layer.lines.pop(0)
        if layer.z_height in z_hop_layers:								# Set z hop in that layer to True if it has a known hop.
            layer.has_z_hop = True
    model = Model(layers, max(xs), max(ys), max(zs), min(xs), min(ys), min(zs), hop)
    return model

def reprocess_model(model, data, z_offset, manager, identifier):
	""" Function for recreating sample points from base_model. Called in reprocess_gcode in _init_.py

    Parameters
    ----------
    model : Model
            Base model to process
	data : dict
            Stores layer spacing and method data.

			data['Spacing'] : int
				Sampling frequency
			data['Random Sampling'] : int
				Number of points to sample using Random Sampling
			data['Min-max'] : int
				Number of points to sample using Min-max
			data['Inside-outside'] : int
				Number of points to sample using Inside-outside
			data['Shift-samples'] : bool
				Whether or not to sample points off the boundary for layer shifting
    """
	layers = model.layers
	points_generated = "points_not_generated"
	message_cooldown = 1			# Time in seconds to wait before sending another layer update message
	last_time = 0					# Time in which last message update was sent
	for i in range(len(layers)):
		layers[i].sample_points['x'] = []
		layers[i].sample_points['y'] = []
		layers[i].sample_points['material'] = []

		if(i % int(data["Spacing"])) == 0 and layers[i].z_height > layers[i].probe_reach: #Sample only layers above probe's reach to prevent false readings from build plate interference
			points_generated = "points_generated_successfully"
			layers[i].gen_sample_points(data, layers, i, z_offset)	#Generate sample points for desired layers
			if time.time() - last_time > message_cooldown:
				manager.send_plugin_message(identifier, dict(message=str(round((i / len(layers)) * 100)) + "% Complete"))
				last_time = time.time()
	return points_generated
#	new_model = Model(layers, model.max_x, model.max_y, model.max_z, model.min_x, model.min_y, model.min_z)
#	return new_model

def remove_last_indices(array, count):
	for _ in range(count):
		array.pop()
	return array


