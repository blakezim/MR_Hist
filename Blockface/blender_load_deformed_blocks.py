import bpy
import glob

block_list = sorted(glob.glob('/hdscratch/ucair/18_062/blockface/block*'))

for block in block_list:
    block_name = block.split('/')[-1]
    file_loc = '/hdscratch/ucair/18_062/blockface/{0}/surfaces/deformable/{0}_ext_deformable.obj'.format(block_name)
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
