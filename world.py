import bpy
import json
import time
import operator
from math import radians, degrees
import mathutils
import os
import yaml

"""
TODO - LOAD PARAMETERS FROM YAML TO DETERMINE TRAIN/T/V SPLIT ETC
"""

P = yaml.load('parameters.txt')


# Delete default cube
bpy.ops.object.select_all(action='DESELECT')
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

P['nT'] = 60
out_dir = os.path.join(os.getcwd(), 'images')


# Style stuff
bpy.context.scene.render.use_freestyle = True
bpy.context.scene.render.line_thickness = 4

bpy.context.scene.render.resolution_y = 64
bpy.context.scene.render.resolution_x = 64

bpy.context.scene.render.resolution_percentage =100
bpy.data.worlds["World"].horizon_color = (0,0,0)
bpy.data.materials.new(name='mat')
bpy.data.materials['mat'].diffuse_color = ((1,1,1))
bpy.data.materials['mat'].specular_color = ((1,1,1))
bpy.data.materials['mat'].translucency = 1.0
bpy.data.materials['mat'].emit =  1
bpy.data.materials['mat'].line_color = ((1,1,1,1))

bpy.data.materials['mat'].use_shadeless = True

bpy.data.lamps['Lamp'].shadow_method = 'NOSHADOW'

bpy.data.scenes["Scene"].render.image_settings.color_mode = 'RGB'
bpy.data.scenes["Scene"].render.line_thickness = 0.5

# Define Shapes
primitives = {'Cube'     : lambda: bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0), radius=1),
              'Icosphere': lambda: bpy.ops.mesh.primitive_ico_sphere_add(location=(0.0, 0.0, 0.0), size=2.5),
              'Cylinder' : lambda: bpy.ops.mesh.primitive_cylinder_add(location=(0.0, 0.0, 0.0), radius=1),
              'Cone'     : lambda: bpy.ops.mesh.primitive_cone_add(location=(0.0, 0.0, 0.0), radius1=1, radius2=0),
              #'Sphere'   : lambda: bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, location=(0.0, 0.0, 0.0), size=2.5),
              'Torus'    : lambda: bpy.ops.mesh.primitive_torus_add(location=(0.0, 0.0, 0.0)),
              #'Monkey'   : lambda: bpy.ops.mesh.primitive_monkey_add(location=(0.0, 0.0, 0.0))
          }
shapes = {}
for item in primitives.keys():
    if item in P['shapes']:
        shapes[item] = primitives[item]

col = {'r':(1,0,0), 'g':(0,1,0), 'b':(0,0,1), 'w':(1,1,1)}
colours = {}
for item in col.keys():
    if item in P['colours']:
        colours[item] = col[item]

locations = {'left' :(1,0,0), 'right' : (-1,0,0),
             'up' :(0,0,1), 'down' : (0,0,-1),
             'upleft':(1,0,1), 'downright': (-1,0,-1),
             'downleft':(1,0,-1),'upright': (-1,0,1), 'c':(0,0,0)}

# Define rotations by axes
rotations = {'x' : (1,0,0), 'y' : (0,1,0), 'z' : (0,0,1),
             'xy': (1,1,0), 'xz': (1,0,1), 'zy': (0,1,1), 'xyz':(1,1,1)}

scales = [1,1.5,2,2.5]
scj = bpy.context.scene

# Iter geometry
for shape, sh_f in shapes.items():
    for col, col_f in colours.items():
        for scale_count, scale in enumerate(scales):
            for locname, loc in locations.items():
                # Create shape
                exitStatus = sh_f()

                ob = bpy.context.object

                me = ob.data

                ob.name = 'obj'
                data_dicts = []

                ob.location = loc

                ob.scale = (scale,scale,scale)
                mat = bpy.data.materials.get('mat')
                mat.diffuse_color = col_f
                me.materials.append(mat)
                if shape in ['Sphere', 'Icosphere']:
                    ob.modifiers.new(name='Wireframe', type='WIREFRAME')
                    ob.modifiers["Wireframe"].thickness=0.005 # parameter initialization
                    ob.modifiers['Wireframe'].use_replace = False
                    ob.modifiers['Wireframe'].use_relative_offset = True
                    ob.scale = (scale-0.8,scale-0.8,scale-0.8)

                # Iter dynamics
                for key, value in rotations.items():

                    verts  = []
                    previous_rotation = [0.0,0.0,0.0]

                    # Iter time
                    for i in range(P['nT']):

                        # Rotate
                        ob.rotation_euler =  [ax * radians((360//P['nT']) * i) for ax in value]

                        # Render
                        scj.render.filepath = os.path.join(out_dir, shape, col, 's'+str(scale_count), locname, key, '{}.png'.format(str(i).zfill(3)) )
                        bpy.ops.render.render( write_still=True )

                        # Get new point cloud - global coordinates
                        #coordinates = [(ob.matrix_world * c.co) for c in ob.data.vertices]
                        #verts = [[v_ for v_ in v ] for v in coordinates]

                        # Get new angles and velocity - assume local rotation (object-centric)
                        #rotation = [degrees(r) for r in bpy.context.active_object.rotation_euler]
                        #rotvel   = list(map(operator.sub, rotation, previous_rotation))
                        #previous_rotation = rotation

                    # JSON structure - could be better
                    data = {'shape': shape, 'rotation_axis':key, 'colour':col}
                            #'data': {'t':i+1,
                            #'local_rotation':rotation, 'local_rotvel':rotvel,
                            #'num_verts': len(ob.data.vertices),'point_cloud':verts}}

                    data_dicts.append(data)

                    obj_filepath = os.path.join(out_dir, shape, col, 's'+str(scale_count), locname, key,'obj.json')
                    with open(obj_filepath, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(data_dicts))

                # Remove object from scene
                objs = bpy.data.objects
                objs.remove(objs[ob.name],True)
