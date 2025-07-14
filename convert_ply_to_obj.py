import trimesh

mesh = trimesh.load('demo_data/ycbv/models/obj_000009.ply')

# Scale from mm to meters
mesh.apply_scale(0.001)

mesh.export('demo_data/spam_potted_meat_can/mesh/converted_obj_000009.obj')